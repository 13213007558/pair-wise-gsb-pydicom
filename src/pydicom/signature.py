# Copyright 2008-2025 pydicom authors. See LICENSE file for details.
"""Signature-aware edit planning for DICOM datasets.

A dataset may contain *Digital Signatures Sequence* (FFFA,FFFA) items that
sign a subset of the dataset's elements. The covered elements are listed in
*Data Elements Signed* (0400,0020), either on the signature item itself or,
if absent there, on the *MAC Parameters Sequence* (4FFE,0001) item
referenced through *MAC ID Number* (0400,0005). Editing a covered element
invalidates the corresponding signature, but a plain
:class:`~pydicom.dataset.Dataset` assignment gives no warning, so stale
signatures are easily saved along
with modified data and downstream readers may believe the content is still
protected.

This module implements an edit plan that stages the intended changes first,
reports which signatures would be invalidated, and then - at the caller's
choice - either refuses to apply the changes or removes the affected
signatures before applying them. Each sequence item dataset is treated as
its own signature scope, so nested *Digital Signatures Sequence* elements
are evaluated against the changes inside their own item.

Certificate trust verification and re-signing are out of scope.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydicom.datadict import dictionary_VR, keyword_for_tag
from pydicom.dataelem import DataElement
from pydicom.tag import BaseTag, Tag

if TYPE_CHECKING:
    from pydicom.dataset import Dataset

DIGITAL_SIGNATURES_SEQUENCE = Tag(0xFFFA, 0xFFFA)
MAC_PARAMETERS_SEQUENCE = Tag(0x4FFE, 0x0001)
DATA_ELEMENTS_SIGNED = Tag(0x0400, 0x0020)
MAC_ID_NUMBER = Tag(0x0400, 0x0005)
DIGITAL_SIGNATURE_UID = Tag(0x0400, 0x0100)

# A path component: (element tag, item index). The index is None for the
# final component, which addresses the element itself rather than descending
# into a sequence item.
_PathComponent = tuple[BaseTag, "int | None"]


class SignatureInvalidationError(ValueError):
    """Raised when planned changes would invalidate digital signatures.

    Carries the affected signatures in :attr:`invalidated`.
    """

    def __init__(self, invalidated: list["InvalidatedSignature"]) -> None:
        self.invalidated = list(invalidated)
        details = "; ".join(
            f"signature {inv.uid or '<no UID>'} in {inv.scope_description} "
            f"covers {sorted(keyword_for_tag(t) or str(t) for t in inv.tags)}"
            for inv in self.invalidated
        )
        super().__init__(
            f"The planned changes would invalidate {len(self.invalidated)} "
            f"digital signature(s): {details}. Use policy='remove' to discard "
            "the affected signatures, or change the planned elements."
        )


@dataclass
class InvalidatedSignature:
    """A digital signature that would be invalidated by a planned change.

    Attributes
    ----------
    scope : Dataset
        The dataset that directly contains the *Digital Signatures
        Sequence* the signature belongs to.
    item : Dataset
        The signature item within the *Digital Signatures Sequence*.
    index : int
        The index of `item` within the *Digital Signatures Sequence*.
    scope_path : tuple of (BaseTag, int)
        The sequence item path from the plan's root dataset to `scope`;
        empty if `scope` is the root dataset itself.
    tags : set of BaseTag
        The planned-change tags covered by this signature.
    """

    scope: "Dataset"
    item: "Dataset"
    index: int
    scope_path: tuple[tuple[BaseTag, int], ...]
    tags: set[BaseTag] = field(default_factory=set)

    @property
    def uid(self) -> "str | None":
        """The *Digital Signature UID* of the signature, if present."""
        if DIGITAL_SIGNATURE_UID in self.item:
            return str(self.item[DIGITAL_SIGNATURE_UID].value)
        return None

    @property
    def scope_description(self) -> str:
        """A human-readable location of the signature's scope."""
        if not self.scope_path:
            return "top-level dataset"
        return " > ".join(
            f"{keyword_for_tag(tag) or tag}[{index}]"
            for tag, index in self.scope_path
        )


@dataclass
class _Change:
    kind: str  # "set" or "delete"
    components: tuple[_PathComponent, ...]
    value: Any = None
    vr: "str | None" = None


def _is_tag_tuple(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    )


def _parse_path(path: Any) -> tuple[_PathComponent, ...]:
    """Normalize an element path to a tuple of (tag, index) components.

    A path is either a single tag/keyword or a flat sequence alternating
    tags and item indices, e.g. ``("ReferencedSeriesSequence", 0,
    "StudyInstanceUID")``. A 2-tuple of two ints is always interpreted as
    a single (group, element) tag.
    """
    if isinstance(path, (str, int, BaseTag)) or _is_tag_tuple(path):
        return ((Tag(path), None),)

    if not isinstance(path, (list, tuple)) or not path:
        raise TypeError(
            f"Invalid element path {path!r}; expected a tag, keyword or a "
            "sequence of path components"
        )

    items = list(path)
    components: list[_PathComponent] = []
    pos = 0
    while pos < len(items):
        item = items[pos]
        if not (isinstance(item, (str, int, BaseTag)) or _is_tag_tuple(item)):
            raise TypeError(f"Invalid path component {item!r}")
        if (
            pos + 1 < len(items)
            and isinstance(items[pos + 1], int)
            and not isinstance(items[pos + 1], bool)
        ):
            components.append((Tag(item), items[pos + 1]))
            pos += 2
        else:
            components.append((Tag(item), None))
            pos += 1

    for tag, index in components[:-1]:
        if index is None:
            raise ValueError(
                "Only the last path component may be an element tag "
                "without an item index"
            )
    if components[-1][1] is not None:
        raise ValueError(
            "The last path component must address an element, not "
            "a sequence item"
        )

    return tuple(components)


def _resolve_path(
    root: "Dataset", components: tuple[_PathComponent, ...]
) -> list[tuple["Dataset", BaseTag]]:
    """Return the (scope dataset, touched tag) pair for each path level."""
    levels: list[tuple["Dataset", BaseTag]] = []
    ds = root
    for tag, index in components:
        levels.append((ds, tag))
        if index is None:
            continue
        name = keyword_for_tag(tag) or str(tag)
        if tag not in ds:
            raise ValueError(f"Path element {name} is not present in the dataset")
        elem = ds[tag]
        if elem.VR != "SQ":
            raise ValueError(f"Path element {name} is not a sequence")
        if not 0 <= index < len(elem.value):
            raise ValueError(
                f"Item index {index} is out of range for {name} "
                f"with {len(elem.value)} item(s)"
            )
        ds = elem.value[index]
    return levels


def _tag_set(value: Any) -> set[BaseTag]:
    if isinstance(value, (BaseTag, int)):
        return {Tag(value)}
    return {Tag(v) for v in value}


def _signed_tags(scope: "Dataset", signature_item: "Dataset") -> set[BaseTag]:
    """Return the tags covered by a signature item within its scope.

    Coverage is given by the item's own *Data Elements Signed* if present,
    otherwise by the *Data Elements Signed* of the *MAC Parameters
    Sequence* item referenced through *MAC ID Number*.
    """
    if DATA_ELEMENTS_SIGNED in signature_item:
        return _tag_set(signature_item[DATA_ELEMENTS_SIGNED].value)

    tags: set[BaseTag] = set()
    if MAC_ID_NUMBER in signature_item and MAC_PARAMETERS_SEQUENCE in scope:
        mac_id = signature_item[MAC_ID_NUMBER].value
        for mac_item in scope[MAC_PARAMETERS_SEQUENCE].value:
            if MAC_ID_NUMBER not in mac_item or mac_item[MAC_ID_NUMBER].value != mac_id:
                continue
            if DATA_ELEMENTS_SIGNED in mac_item:
                tags |= _tag_set(mac_item[DATA_ELEMENTS_SIGNED].value)

    return tags


def _prune_mac_parameters(scope: "Dataset") -> None:
    """Remove MAC parameter items no longer referenced by any signature."""
    if MAC_PARAMETERS_SEQUENCE not in scope:
        return

    referenced: set[Any] = set()
    if DIGITAL_SIGNATURES_SEQUENCE in scope:
        for item in scope[DIGITAL_SIGNATURES_SEQUENCE].value:
            if MAC_ID_NUMBER in item:
                referenced.add(item[MAC_ID_NUMBER].value)

    elem = scope[MAC_PARAMETERS_SEQUENCE]
    for index in range(len(elem.value) - 1, -1, -1):
        mac_item = elem.value[index]
        if MAC_ID_NUMBER in mac_item and (
            mac_item[MAC_ID_NUMBER].value not in referenced
        ):
            del elem.value[index]

    if not elem.value:
        del scope[MAC_PARAMETERS_SEQUENCE]


class SignatureEditPlan:
    """A signature-aware plan of changes to a :class:`~pydicom.dataset.Dataset`.

    Stage changes with :meth:`set` and :meth:`delete`, inspect
    :attr:`invalidated_signatures` to see which digital signatures the
    changes would invalidate, then call :meth:`apply` with the desired
    policy. Created via :meth:`Dataset.signature_edit_plan
    <pydicom.dataset.Dataset.signature_edit_plan>`.
    """

    def __init__(self, dataset: "Dataset") -> None:
        self._dataset = dataset
        self._changes: list[_Change] = []

    @property
    def dataset(self) -> "Dataset":
        """The dataset the planned changes apply to."""
        return self._dataset

    def set(
        self, path: Any, value: Any, VR: "str | None" = None
    ) -> "SignatureEditPlan":
        """Stage setting the element at `path` to `value`.

        Parameters
        ----------
        path
            The element to change: a keyword or tag, or a path into nested
            sequence items such as
            ``("ReferencedSeriesSequence", 0, "StudyInstanceUID")`` where
            each ``(tag, index)`` component descends into an item of the
            sequence at `tag`.
        value
            The new value. May be a :class:`~pydicom.dataelem.DataElement`
            or a plain value. For existing elements the value is converted
            using the element's VR; for new elements the VR is taken from
            the data dictionary unless `VR` is given.
        VR : str, optional
            The value representation to use when adding a new element that
            is not in the DICOM data dictionary (e.g. a private tag).

        Returns
        -------
        SignatureEditPlan
            The plan itself, to allow chaining.
        """
        self._changes.append(_Change("set", _parse_path(path), value, VR))
        return self

    def delete(self, path: Any) -> "SignatureEditPlan":
        """Stage deleting the element at `path`.

        Parameters
        ----------
        path
            The element to delete, in the same form as for :meth:`set`.

        Returns
        -------
        SignatureEditPlan
            The plan itself, to allow chaining.
        """
        self._changes.append(_Change("delete", _parse_path(path)))
        return self

    def clear(self) -> None:
        """Discard all staged changes."""
        self._changes.clear()

    @property
    def invalidated_signatures(self) -> list[InvalidatedSignature]:
        """The signatures that would be invalidated by the staged changes.

        A signature is invalidated if a changed tag is listed in its
        *Data Elements Signed* (directly or via the referenced *MAC
        Parameters Sequence* item), or if the change descends into a
        sequence whose tag is covered. Nested item datasets are evaluated
        against their own *Digital Signatures Sequence*.
        """
        found: dict[tuple[int, int], InvalidatedSignature] = {}
        for change in self._changes:
            levels = _resolve_path(self._dataset, change.components)
            for depth, (scope, tag) in enumerate(levels):
                if DIGITAL_SIGNATURES_SEQUENCE not in scope:
                    continue
                for index, item in enumerate(
                    scope[DIGITAL_SIGNATURES_SEQUENCE].value
                ):
                    if tag not in _signed_tags(scope, item):
                        continue
                    key = (id(scope), index)
                    if key not in found:
                        scope_path = tuple(
                            (t, i)  # type: ignore[misc]
                            for t, i in change.components[:depth]
                        )
                        found[key] = InvalidatedSignature(
                            scope=scope,
                            item=item,
                            index=index,
                            scope_path=scope_path,
                        )
                    found[key].tags.add(tag)
        return list(found.values())

    def apply(self, policy: str = "remove") -> list[InvalidatedSignature]:
        """Apply the staged changes according to `policy`.

        Parameters
        ----------
        policy : str
            ``"reject"`` raises :class:`SignatureInvalidationError` without
            changing anything if any signature would be invalidated.
            ``"remove"`` (default) removes the invalidated signatures from
            their *Digital Signatures Sequence* - including any *MAC
            Parameters Sequence* items no longer referenced by a remaining
            signature - and then applies the changes.

        Returns
        -------
        list of InvalidatedSignature
            The signatures that were removed (empty with ``"reject"``).

        Raises
        ------
        SignatureInvalidationError
            If `policy` is ``"reject"`` and the changes would invalidate
            at least one signature.
        ValueError
            If `policy` is unknown, a path cannot be resolved, or the VR of
            a new element cannot be determined.
        """
        if policy not in ("remove", "reject"):
            raise ValueError(
                f"Unknown policy {policy!r}, expected 'remove' or 'reject'"
            )

        invalidated = self.invalidated_signatures
        if policy == "reject" and invalidated:
            raise SignatureInvalidationError(invalidated)

        if invalidated:
            self._remove_signatures(invalidated)

        for change in self._changes:
            self._apply_change(change)

        self._changes.clear()
        return invalidated

    @staticmethod
    def _remove_signatures(invalidated: list[InvalidatedSignature]) -> None:
        by_scope: dict[int, tuple["Dataset", set[int]]] = {}
        for inv in invalidated:
            _, indices = by_scope.setdefault(id(inv.scope), (inv.scope, set()))
            indices.add(inv.index)

        for scope, indices in by_scope.values():
            elem = scope[DIGITAL_SIGNATURES_SEQUENCE]
            for index in sorted(indices, reverse=True):
                del elem.value[index]
            if not elem.value:
                del scope[DIGITAL_SIGNATURES_SEQUENCE]
            _prune_mac_parameters(scope)

    def _apply_change(self, change: _Change) -> None:
        levels = _resolve_path(self._dataset, change.components)
        scope, tag = levels[-1]

        if change.kind == "delete":
            del scope[tag]
            return

        value, vr = change.value, change.vr
        if isinstance(value, DataElement):
            scope[tag] = value
        elif tag in scope:
            scope[tag].value = value
        else:
            if vr is None:
                try:
                    vr = dictionary_VR(tag)
                except KeyError:
                    raise ValueError(
                        f"Unable to determine the VR for new element {tag}; "
                        "pass it explicitly using the 'VR' argument"
                    ) from None
            scope.add_new(tag, vr, value)
