"""Batch UID remapping engine built on pydicom.

Workflow for one batch:

1. Scan every input file and collect each UI element occurrence,
   classified as instance / definition / unknown.
2. Build the old->new map: caller-supplied entries (mapping_in) win;
   every other distinct old instance UID is passed to the caller's
   assign function.
3. Validate the whole plan BEFORE touching anything: unstable assign
   functions, malformed or over-long new UIDs, and collisions are all
   reported together in one RemapValidationError and nothing is written.
4. Apply the map recursively (sequences included), force the File Meta
   MediaStorageSOPInstanceUID to match the dataset SOPInstanceUID, and
   write the files plus change / unclassified reports and the merged
   mapping for the next batch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ._pydicom import pydicom
from .uidtags import classify

_UID_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")
_UID_MAX_LEN = 64

SOP_INSTANCE_UID = (0x0008, 0x0018)
MEDIA_STORAGE_SOP_INSTANCE_UID = (0x0002, 0x0003)


class RemapValidationError(Exception):
    """Raised before any file is modified when the plan is unsafe."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("UID remap validation failed:\n  " + "\n  ".join(problems))


@dataclass
class UiOccurrence:
    file: str
    path: str
    tag: tuple[int, int]
    kind: str            # instance | definition | unknown
    values: list[str]    # multi-valued UI components


@dataclass
class Change:
    file: str
    path: str
    keyword: str
    old: str
    new: str


@dataclass
class RemapResult:
    mapping: dict[str, str]
    changes: list[Change]
    unclassified: list[dict]
    warnings: list[str]
    output_files: list[str]

    def report(self) -> dict:
        return {
            "mapping": self.mapping,
            "changes": [vars(c) for c in self.changes],
            "unclassified_private_ui": self.unclassified,
            "warnings": self.warnings,
            "output_files": self.output_files,
        }


def is_valid_uid(uid: str) -> bool:
    if not isinstance(uid, str) or not uid or len(uid) > _UID_MAX_LEN:
        return False
    if not _UID_RE.match(uid):
        return False
    # DICOM forbids leading zeros in a component unless it is exactly "0".
    return all(part == "0" or not part.startswith("0") for part in uid.split("."))


def _tag_pair(tag) -> tuple[int, int]:
    return (int(tag) >> 16, int(tag) & 0xFFFF)


def _display_name(tag) -> str:
    keyword = pydicom.datadict.keyword_for_tag(tag)
    if keyword:
        return keyword
    pair = _tag_pair(tag)
    return f"Tag{pair[0]:04X},{pair[1]:04X}"


def _ui_values(element) -> list[str]:
    value = element.value
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _iter_ui(dataset, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str, object]]:
    for element in dataset:
        name = _display_name(element.tag)
        if element.VR == "SQ":
            for index, item in enumerate(element.value):
                yield from _iter_ui(item, path + (f"{name}[{index}]",))
        elif element.VR == "UI":
            yield path, name, element


def _scan_file(name: str, ds) -> list[UiOccurrence]:
    occurrences: list[UiOccurrence] = []
    scopes = [("FileMeta", ds.file_meta)] if ds.file_meta is not None else []
    scopes.append(("Dataset", ds))
    for scope, container in scopes:
        for path, elem_name, element in _iter_ui(container):
            tag_path = "/".join((scope,) + path + (elem_name,))
            occurrences.append(UiOccurrence(
                file=name, path=tag_path, tag=_tag_pair(element.tag),
                kind=classify(_tag_pair(element.tag)), values=_ui_values(element),
            ))
    return occurrences


def build_mapping(
    old_uids: Iterable[str],
    assign: Callable[[str], str],
    mapping_in: dict[str, str] | None = None,
    definition_uids: Iterable[str] = (),
) -> dict[str, str]:
    """Resolve every old UID to a new UID, validating the whole plan.

    Raises RemapValidationError listing every problem found; nothing is
    applied when validation fails.
    """
    mapping_in = dict(mapping_in or {})
    problems: list[str] = []
    mapping: dict[str, str] = {}

    for old in sorted(set(old_uids)):
        if old in mapping_in:
            new = mapping_in[old]
        else:
            new = assign(old)
            again = assign(old)
            if new != again:
                problems.append(
                    f"unstable mapping: assign({old!r}) returned {new!r} then {again!r}")
        mapping[old] = new

    inverse: dict[str, str] = {}
    old_set = set(mapping)
    definition_set = set(definition_uids)
    for old, new in mapping.items():
        if not is_valid_uid(new):
            problems.append(
                f"invalid new UID for {old!r}: {new!r} "
                f"(must be digits/dots, no leading zeros, <= {_UID_MAX_LEN} chars)")
            continue
        if new in inverse and inverse[new] != old:
            problems.append(
                f"conflict: {old!r} and {inverse[new]!r} both map to {new!r}")
        inverse.setdefault(new, old)
        if new in old_set and new != old:
            problems.append(
                f"conflict: new UID {new!r} for {old!r} collides with another "
                f"existing instance UID in this batch")
        if new in definition_set:
            problems.append(
                f"conflict: new UID {new!r} for {old!r} collides with a "
                f"definition-class UID present in the batch")
    if problems:
        raise RemapValidationError(problems)
    return mapping


def _apply(dataset, scope: str, file_name: str, mapping: dict[str, str],
           changes: list[Change], path: tuple[str, ...] = ()) -> None:
    for element in dataset:
        name = _display_name(element.tag)
        if element.VR == "SQ":
            for index, item in enumerate(element.value):
                _apply(item, scope, file_name, mapping, changes,
                       path + (f"{name}[{index}]",))
        elif element.VR == "UI" and classify(_tag_pair(element.tag)) == "instance":
            old_values = _ui_values(element)
            new_values = [mapping.get(v, v) for v in old_values]
            if new_values == old_values:
                continue
            changes.append(Change(
                file=file_name,
                path="/".join((scope,) + path + (name,)),
                keyword=name,
                old="\\".join(old_values),
                new="\\".join(new_values),
            ))
            if isinstance(element.value, (list, tuple)) or len(new_values) != 1:
                element.value = new_values
            else:
                element.value = new_values[0]


def _sync_file_meta(ds, file_name: str, mapping: dict[str, str],
                    changes: list[Change], warnings: list[str]) -> None:
    if ds.file_meta is None or "SOPInstanceUID" not in ds:
        return
    dataset_uid = str(ds.SOPInstanceUID)
    meta = ds.file_meta
    if "MediaStorageSOPInstanceUID" not in meta:
        meta.MediaStorageSOPInstanceUID = dataset_uid
        changes.append(Change(
            file=file_name, path="FileMeta/MediaStorageSOPInstanceUID",
            keyword="MediaStorageSOPInstanceUID", old="", new=dataset_uid,
        ))
        return
    meta_uid = str(meta.MediaStorageSOPInstanceUID)
    if meta_uid == dataset_uid:
        return
    if mapping.get(meta_uid, meta_uid) != dataset_uid:
        warnings.append(
            f"{file_name}: File Meta MediaStorageSOPInstanceUID ({meta_uid}) "
            f"disagreed with dataset SOPInstanceUID; forced to {dataset_uid}")
    changes.append(Change(
        file=file_name, path="FileMeta/MediaStorageSOPInstanceUID",
        keyword="MediaStorageSOPInstanceUID", old=meta_uid, new=dataset_uid,
    ))
    meta.MediaStorageSOPInstanceUID = dataset_uid


def remap_batch(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    assign: Callable[[str], str],
    mapping_in: dict[str, str] | None = None,
) -> RemapResult:
    """Remap instance UIDs in every input file into output_dir.

    assign: caller-provided old UID -> new UID function, used for any old
    UID not already covered by mapping_in. Must be deterministic; it is
    called twice per UID and any disagreement aborts the batch.
    """
    inputs = [Path(p) for p in inputs]
    output_dir = Path(output_dir)

    loaded = [(p, pydicom.dcmread(str(p))) for p in inputs]

    all_occurrences: list[UiOccurrence] = []
    for path, ds in loaded:
        all_occurrences.extend(_scan_file(path.name, ds))

    old_uids = sorted({v for o in all_occurrences if o.kind == "instance" for v in o.values})
    definition_uids = {v for o in all_occurrences if o.kind == "definition" for v in o.values}
    mapping = build_mapping(old_uids, assign, mapping_in, definition_uids)

    changes: list[Change] = []
    warnings: list[str] = []
    unclassified = [
        {
            "file": o.file,
            "path": o.path,
            "tag": f"({o.tag[0]:04X},{o.tag[1]:04X})",
            "value": "\\".join(o.values),
            "action": "unchanged (no classification rule)",
        }
        for o in all_occurrences
        if o.kind == "unknown"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    for path, ds in loaded:
        if ds.file_meta is not None:
            _apply(ds.file_meta, "FileMeta", path.name, mapping, changes)
        _apply(ds, "Dataset", path.name, mapping, changes)
        _sync_file_meta(ds, path.name, mapping, changes, warnings)
        target = output_dir / path.name
        ds.save_as(str(target))
        output_files.append(str(target))

    return RemapResult(
        mapping=mapping,
        changes=changes,
        unclassified=unclassified,
        warnings=warnings,
        output_files=output_files,
    )


def load_mapping(path: str | Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("mapping file must contain a JSON object old->new")
    return {str(k): str(v) for k, v in data.items()}


def save_mapping(mapping: dict[str, str], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(mapping.items())), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_report(result: RemapResult, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.report(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

