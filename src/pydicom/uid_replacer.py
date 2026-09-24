"""Batch replacement of DICOM instance-identity UIDs.

This module rewrites the UIDs that identify a DICOM *instance* (Study,
Series, SOP Instance, Frame of Reference and the standard sequences that
reference them) while leaving *definitional* UIDs untouched:

* SOP Class UIDs (including Media Storage and referenced SOP classes)
* Transfer Syntax UIDs
* Coding Scheme / Context Group / Mapping Resource UIDs
* Implementation, creator and manufacturer-class UIDs

The caller supplies an allocator(old_uid) -> new_uid callable.  Its result
is cached, so the same old UID always receives the same new UID no matter
how often it appears and no matter which file it appears in.  The mapping
can be exported and fed back into a new UIDRemapper so a later batch
allocates the same new UIDs for previously seen identities.

All validation (duplicate new UIDs, illegal UID format/length, unstable
allocator output) happens before any file is modified.

The module does one thing only: the UID transformation the caller asks for.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydicom import dcmread
from pydicom.dataset import Dataset

#: VR UI elements that describe *what kind of object* this is.
#: These are never rewritten.
DEFINITIONAL_UID_KEYWORDS: frozenset[str] = frozenset(
    {
        "AffectedSOPClassUID",
        "AvailableTransferSyntaxUID",
        "CodingSchemeUID",
        "ContextGroupExtensionCreatorUID",
        "CreatorVersionUID",
        "EncryptedContentTransferSyntaxUID",
        "FlowTransferSyntaxUID",
        "HomeCommunityID",
        "ImplementationClassUID",
        "InstanceCreatorUID",
        "MACCalculationTransferSyntaxUID",
        "ManufacturerDeviceClassUID",
        "MappingResourceUID",
        "MediaStorageSOPClassUID",
        "OriginalSpecializedSOPClassUID",
        "PertinentSOPClassesInSeries",
        "PertinentSOPClassesInStudy",
        "PrivateInformationCreatorUID",
        "ReferencedRelatedGeneralSOPClassUIDInFile",
        "ReferencedSOPClassUID",
        "ReferencedSOPClassUIDInFile",
        "ReferencedTransferSyntaxUIDInFile",
        "RelatedGeneralSOPClassUID",
        "RepositoryUniqueID",
        "RequestedSOPClassUID",
        "RTVCommunicationSOPClassUID",
        "SOPClassUID",
        "SOPClassesInStudy",
        "SOPClassesSupported",
        "StoredInstanceTransferSyntaxUID",
        "TemplateExtensionCreatorUID",
        "TemplateExtensionOrganizationUID",
        "TransferSyntaxUID",
    }
)

#: VR UI elements that identify a DICOM instance (or reference one).
#: These are rewritten through the caller-supplied allocator.
INSTANCE_UID_KEYWORDS: frozenset[str] = frozenset(
    {
        "AcquisitionUID",
        "AffectedSOPInstanceUID",
        "AnnotationGroupUID",
        "AssertionUID",
        "ConcatenationUID",
        "ConceptualVolumeUID",
        "ConstituentConceptualVolumeUID",
        "DimensionOrganizationUID",
        "DoseReferenceUID",
        "DosimetricObjectiveUID",
        "EquipmentFrameOfReferenceUID",
        "FailedSOPInstanceUIDList",
        "FiducialUID",
        "FrameOfReferenceUID",
        "IrradiationEventUID",
        "LargePaletteColorLookupTableUID",
        "MediaStorageSOPInstanceUID",
        "ModelGroupUID",
        "MultiFrameSourceSOPInstanceUID",
        "MultiplexGroupUID",
        "ObservationUID",
        "PaletteColorLookupTableUID",
        "PatientSetupUID",
        "PresentationDisplayCollectionUID",
        "PresentationSequenceCollectionUID",
        "PrivateRecordUID",
        "PyramidUID",
        "RTVCommunicationSOPInstanceUID",
        "RadiopharmaceuticalAdministrationEventUID",
        "ReferencedAssertionUID",
        "ReferencedColorPaletteInstanceUID",
        "ReferencedConceptualVolumeUID",
        "ReferencedContentItem",
        "ReferencedDoseReferenceUID",
        "ReferencedDosimetricObjectiveUID",
        "ReferencedFiducialUID",
        "ReferencedFiducialsUID",
        "ReferencedFrameOfReferenceUID",
        "ReferencedGeneralPurposeScheduledProcedureStepTransactionUID",
        "ReferencedSOPInstanceUID",
        "ReferencedSOPInstanceUIDInFile",
        "ReferencedTreatmentPositionGroupUID",
        "RelatedFrameOfReferenceUID",
        "RequestedSOPInstanceUID",
        "RTTreatmentPhaseUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "SOPInstanceUIDOfConcatenationSource",
        "SourceConceptualVolumeUID",
        "SourceFrameOfReferenceUID",
        "SpecimenUID",
        "StorageMediaFileSetUID",
        "StudyInstanceUID",
        "SynchronizationFrameOfReferenceUID",
        "TableFrameOfReferenceUID",
        "TargetFrameOfReferenceUID",
        "TargetUID",
        "TrackingUID",
        "TransactionUID",
        "TreatmentPositionGroupUID",
        "TreatmentSessionUID",
        "VolumeFrameOfReferenceUID",
        "VolumetricPresentationInputSetUID",
    }
)

INSTANCE = "instance"
DEFINITIONAL = "definitional"
UNCLASSIFIED = "unclassified"

_UID_MAX_LENGTH = 64


def validate_uid(value: str) -> str | None:
    """Return an error message if value is not a legal DICOM UID.

    The maximum UID length is 64 characters and only digits separated by
    single dots are permitted, with no empty or zero-padded components.
    """
    if not isinstance(value, str) or not value:
        return "UID is empty"
    if len(value) > _UID_MAX_LENGTH:
        return (
            f"UID {value!r} is {len(value)} characters long (maximum {_UID_MAX_LENGTH})"
        )
    for component in value.split("."):
        if not component or not component.isdigit():
            return f"UID {value!r} contains an empty or non-numeric component"
        if len(component) > 1 and component[0] == "0":
            return f"UID {value!r} contains a zero-padded component {component!r}"
    return None


def classify_ui_element(tag: int, keyword: str | None) -> str:
    """Classify a VR UI element.

    Returns INSTANCE (rewrite), DEFINITIONAL (keep) or UNCLASSIFIED
    (report, do not touch).  Private UI elements are always unclassified:
    the caller must state the rule for them.
    """
    if _is_private_tag(tag):
        return UNCLASSIFIED
    if keyword in DEFINITIONAL_UID_KEYWORDS:
        return DEFINITIONAL
    if keyword in INSTANCE_UID_KEYWORDS:
        return INSTANCE
    # Pattern fallback so newly introduced dictionary entries are handled
    # predictably; definitional patterns win over instance patterns.
    if keyword and (
        keyword.endswith("ClassUID")
        or "TransferSyntax" in keyword
        or "CodingScheme" in keyword
        or "ContextGroup" in keyword
    ):
        return DEFINITIONAL
    if keyword and ("InstanceUID" in keyword or "FrameOfReferenceUID" in keyword):
        return INSTANCE
    return UNCLASSIFIED


def _is_private_tag(tag: int) -> bool:
    return bool((tag >> 16) % 2)


class UIDRemapError(Exception):
    """Raised when a remap plan cannot be validated.

    No file is modified when this exception is raised.  Every detected
    problem is collected in the problems attribute.
    """

    def __init__(self, problems: Iterable[str]):
        self.problems: list[str] = list(problems)
        super().__init__(
            "UID remap rejected before modifying any files:\n"
            + "\n".join(f"  - {problem}" for problem in self.problems)
        )


@dataclass
class UIDChange:
    """One rewritten UI value."""

    filename: str
    path: str
    keyword: str
    old: str
    new: str


@dataclass
class UnclassifiedUI:
    """A UI element whose role is not known.

    The element is left unchanged; the caller decides the rule later.
    """

    filename: str
    path: str
    tag: str
    keyword: str
    value: str


@dataclass
class RemapReport:
    """Result of UIDRemapper.remap."""

    mapping: dict[str, str]
    changes: list[UIDChange] = field(default_factory=list)
    unclassified: list[UnclassifiedUI] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def export_mapping(self, path: str | Path) -> None:
        """Write the old -> new UID mapping as JSON for the next batch."""
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump({"old_to_new": self.mapping}, f, indent=2, sort_keys=True)
            f.write("\n")


def load_mapping(path: str | Path) -> dict[str, str]:
    """Load a mapping written by RemapReport.export_mapping."""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    mapping = data["old_to_new"]
    if not isinstance(mapping, dict):
        raise ValueError(f"Mapping file {path} does not contain an old_to_new object")
    return {str(old): str(new) for old, new in mapping.items()}


def _format_tag(tag: int) -> str:
    return f"({tag >> 16:04X},{tag & 0xFFFF:04X})"


def _keyword_of(tag: int) -> str:
    from pydicom.datadict import keyword_for_tag

    return keyword_for_tag(tag) or "<unknown>"


def _iter_ui_values(
    ds: Dataset, prefix: str = ""
) -> Iterable[tuple[Any, int, int | None, str, str]]:
    """Yield (element, tag, value-index, keyword, path) for each UI value.

    Sequences are recursed into, including their item datasets.
    Multi-valued UI elements are yielded once per component with its index.
    """
    for element in ds:
        tag = int(element.tag)
        path = f"{prefix}{_format_tag(tag)}"
        if element.VR == "SQ":
            for index, item in enumerate(element.value or []):
                yield from _iter_ui_values(item, f"{path}[{index}] > ")
            continue
        if element.VR != "UI":
            continue
        values = element.value
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            for index, value in enumerate(values):
                if value:
                    yield element, tag, index, _keyword_of(tag), f"{path}[{index}]"
        elif values:
            yield element, tag, None, _keyword_of(tag), path


class UIDRemapper:
    """Plan and apply instance-UID replacement across DICOM files.

    Parameters
    ----------
    allocator:
        Callable taking an old UID and returning its new UID.  It must be a
        pure function: it is called twice for each new UID and the two
        answers must agree.  Stateful allocators should pre-generate their
        values and pass them as seed_mapping instead.
    seed_mapping:
        Old -> new UID mapping from a previous batch.  UIDs already in the
        seed are not offered to the allocator, giving stable identities
        across batches.
    """

    def __init__(
        self,
        allocator: Callable[[str], str] | None = None,
        seed_mapping: dict[str, str] | None = None,
    ) -> None:
        self._allocator = allocator
        self._mapping: dict[str, str] = dict(seed_mapping or {})

    @property
    def mapping(self) -> dict[str, str]:
        """Accumulated old -> new mapping (including seeded entries)."""
        return dict(self._mapping)

    def build_mapping(self, old_uids: Iterable[str]) -> dict[str, str]:
        """Allocate new UIDs and validate the complete mapping.

        Raises UIDRemapError collecting every conflict, illegal UID or
        unstable allocator answer found.
        """
        unique_uids = sorted({uid for uid in old_uids if uid})
        uid_set = set(unique_uids)
        unstable: list[str] = []
        for old in unique_uids:
            if old in self._mapping or self._allocator is None:
                continue
            first = self._allocator(old)
            second = self._allocator(old)
            if first != second:
                unstable.append(old)
                continue
            self._mapping[old] = first

        problems: list[str] = [
            f"allocator returned different UIDs on repeated calls for {old!r}"
            for old in unstable
        ]

        relevant = {old: new for old, new in self._mapping.items() if old in uid_set}
        for old, new in relevant.items():
            error = validate_uid(new)
            if error:
                problems.append(f"new UID for {old!r}: {error}")

        # The mapping must be injective: two old identities must not merge.
        seen: dict[str, str] = {}
        for old, new in relevant.items():
            other = seen.get(new)
            if other is not None and other != old:
                problems.append(
                    f"new UID {new!r} assigned to both {other!r} and {old!r}"
                )
            seen.setdefault(new, old)

        # A new UID must not take over a different old identity still in
        # the batch (that would leave references pointing the wrong way).
        for old, new in relevant.items():
            if new in uid_set and self._mapping.get(new) != new:
                problems.append(
                    f"new UID for {old!r} ({new!r}) collides with an existing old UID"
                )

        if self._allocator is None:
            problems.extend(
                f"no allocator supplied and {old!r} is not in the seed mapping"
                for old in unique_uids
                if old not in self._mapping
            )

        if problems:
            raise UIDRemapError(problems)
        return dict(self._mapping)

    def remap(
        self,
        paths: Iterable[str | Path],
        output_dir: str | Path | None = None,
        *,
        in_place: bool = False,
    ) -> RemapReport:
        """Rewrite instance UIDs in the given DICOM files.

        All files are read and the whole mapping is validated before
        anything is written.  Files are written into *output_dir* keeping
        their basenames, or overwritten in place when *in_place* is True.

        Returns a RemapReport with the mapping, a per-element change list
        and the private/unknown UI fields that were left untouched.
        """
        file_paths = [Path(p) for p in paths]
        if not in_place:
            if output_dir is None:
                raise ValueError("provide output_dir or pass in_place=True")
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        # Read everything first so validation errors leave no half-written
        # output behind.
        loaded: list[tuple[Path, Dataset]] = [
            (path, dcmread(path)) for path in file_paths
        ]

        old_uids: set[str] = set()
        for _, ds in loaded:
            old_uids.update(self._collect_instance_uids(ds))

        mapping = self.build_mapping(old_uids)

        report = RemapReport(mapping=mapping)
        for path, ds in loaded:
            name = path.name
            if ds.file_meta is not None:
                self._check_meta_sync(name, ds, report)
            self._rewrite_dataset(name, ds, mapping, report)
            target = path if in_place else out_dir / path.name
            ds.save_as(target, enforce_file_format=False)

        return report

    @staticmethod
    def _collect_instance_uids(ds: Dataset) -> set[str]:
        found: set[str] = set()
        sources = [ds]
        if ds.file_meta is not None:
            sources.append(ds.file_meta)
        for source in sources:
            for element, tag, index, _, _ in _iter_ui_values(source):
                keyword = _keyword_of(tag)
                if classify_ui_element(tag, keyword) == INSTANCE:
                    values = element.value
                    if index is not None:
                        found.add(str(values[index]))
                    elif isinstance(values, (list, tuple)):
                        found.update(str(v) for v in values if v)
                    else:
                        found.add(str(values))
        return found

    def _rewrite_dataset(
        self,
        filename: str,
        ds: Dataset,
        mapping: dict[str, str],
        report: RemapReport,
    ) -> None:
        sources = [(ds, "")]
        if ds.file_meta is not None:
            sources.append((ds.file_meta, "File Meta > "))
        for source, source_prefix in sources:
            for element, tag, index, keyword, path in _iter_ui_values(
                source, source_prefix
            ):
                old = element.value if index is None else element.value[index]
                kind = classify_ui_element(tag, keyword)
                if kind == DEFINITIONAL:
                    continue
                if kind == UNCLASSIFIED:
                    report.unclassified.append(
                        UnclassifiedUI(
                            filename=filename,
                            path=path,
                            tag=_format_tag(tag),
                            keyword=keyword,
                            value=str(old),
                        )
                    )
                    continue
                new = mapping[str(old)]
                if new != old:
                    report.changes.append(
                        UIDChange(
                            filename=filename,
                            path=path,
                            keyword=keyword,
                            old=str(old),
                            new=new,
                        )
                    )
                if index is None:
                    element.value = new
                else:
                    element.value[index] = new

    @staticmethod
    def _check_meta_sync(filename: str, ds: Dataset, report: RemapReport) -> None:
        meta_uid = getattr(ds.file_meta, "MediaStorageSOPInstanceUID", None)
        body_uid = getattr(ds, "SOPInstanceUID", None)
        if meta_uid is not None and body_uid is not None and meta_uid != body_uid:
            report.warnings.append(
                f"{filename}: File Meta MediaStorageSOPInstanceUID "
                f"({meta_uid}) differs from SOPInstanceUID ({body_uid}); "
                "each value is remapped independently"
            )
