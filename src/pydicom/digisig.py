# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Signature-aware edit planning for DICOM datasets.

DICOM digital signatures (PS3.5 Annex C.7.6) are stored in the
*Digital Signatures Sequence* ``(FFFA,FFFA)`` of a dataset. Each item holds
a *MAC Parameters Sequence* ``(4FFE,0001)`` whose *Data Elements Signed*
``(0400,0020)`` lists the tags covered by that signature. Modifying a covered
element silently invalidates the corresponding signature, which may mislead
downstream consumers into trusting stale content.

This module provides :class:`SignatureEditPlan`, which collects pending
element changes for a dataset (including changes inside nested sequence
items), determines which signatures would be invalidated, and then either
rejects the edit or removes the affected signatures before applying the
changes. Certificate trust verification and re-signing are out of scope.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydicom.dataelem import DataElement, RawDataElement
from pydicom.datadict import keyword_for_tag, tag_for_keyword
from pydicom.tag import BaseTag, Tag

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset

DIGITAL_SIGNATURES_SEQUENCE = BaseTag(0xFFFAFFFA)
MAC_PARAMETERS_SEQUENCE = BaseTag(0x4FFE0001)
DATA_ELEMENTS_SIGNED = BaseTag(0x04000020)
MAC_ID_NUMBER = BaseTag(0x04000005)
DIGITAL_SIGNATURE_UID = BaseTag(0x04000100)

# Path from a plan's root dataset to a nested dataset, as (SQ tag, item index)
_ScopePath = tuple[tuple[BaseTag, int], ...]


class SignatureInvalidationError(Exception):
    """Raised when pending changes would invalidate digital signatures.

    Carries the list of :class:`InvalidatedSignature` describing the
    signatures that the pending changes would invalidate.
    """

    def __init__(self, invalidated: list["InvalidatedSignature"]) -> None:
        self.invalidated = invalidated
        uids = ", ".join(sig.signature_uid or "<no UID>" for sig in invalidated)
        super().__init__(
            f"The pending changes would invalidate {len(invalidated)} "
            f"digital signature(s): {uids}. Commit the plan with "
            "on_invalidated='remove' to discard the affected signatures, "
            "or discard the plan."
        )


@dataclass
class InvalidatedSignature:
    """A digital signature invalidated by one or more pending changes.

    Attributes
    ----------
    scope : Dataset
        The dataset owning the *Digital Signatures Sequence* that contains
        the signature. This is the plan's root dataset for top-level
        signatures, or a sequence item dataset for nested signatures.
    index : int
        Index of the signature item within the scope's
        *Digital Signatures Sequence*.
    signature_uid : str | None
        The *Digital Signature UID* ``(0400,0100)`` of the signature, if
        present.
    mac_id : int | None
        The *MAC ID Number* ``(0400,0005)`` from the signature's
        *MAC Parameters Sequence*, if present.
    data_elements_signed : list of BaseTag
        The tags covered by the signature, from *Data Elements Signed*
        ``(0400,0020)``.
    triggered_by : list of BaseTag
        The pending-change tags that fall within the signature's coverage.
    path : tuple of (BaseTag, int)
        Path from the plan's root dataset to ``scope`` as
        ``(sequence tag, item index)`` pairs; empty for the root dataset.
    """

    scope: "Dataset"
    index: int
    signature_uid: str | None
    mac_id: int | None
    data_elements_signed: list[BaseTag]
    triggered_by: list[BaseTag]
    path: _ScopePath = field(default_factory=tuple)

    def __repr__(self) -> str:
        location = "root dataset"
        if self.path:
            steps = " -> ".join(f"{tag}[{index}]" for tag, index in self.path)
            location = f"nested dataset at {steps}"
        return (
            f"<InvalidatedSignature uid={self.signature_uid!r} "
            f"mac_id={self.mac_id!r} index={self.index} in {location} "
            f"triggered_by={[str(t) for t in self.triggered_by]}>"
        )


def _resolve_tag(key: Any) -> BaseTag:
    """Return the :class:`BaseTag` for an element key (keyword or tag)."""
    if isinstance(key, str):
        tag = tag_for_keyword(key)
        if tag is not None:
            return BaseTag(tag)

    return Tag(key)


def _apply_change(scope: "Dataset", key: Any, value: Any) -> None:
    """Apply a single pending change to ``scope``."""
    if isinstance(value, (DataElement, RawDataElement)):
        scope[_resolve_tag(key)] = value
        return

    if isinstance(key, str):
        setattr(scope, key, value)
        return

    tag = Tag(key)
    if tag in scope:
        scope[tag].value = value
        return

    keyword = keyword_for_tag(tag)
    if keyword:
        setattr(scope, keyword, value)
        return

    raise ValueError(
        f"Unable to set the value of the element {tag}: the element is not "
        "present in the dataset and its VR is unknown, pass a DataElement "
        "as the value instead"
    )


def _signed_tags(signature_item: "Dataset") -> list[BaseTag]:
    """Return the tags covered by a *Digital Signatures Sequence* item."""
    mac_seq = signature_item.get("MACParametersSequence")
    if mac_seq is None:
        return []

    signed = mac_seq[0].get("DataElementsSigned")
    if signed is None:
        return []

    if isinstance(signed, (BaseTag, int)):
        return [BaseTag(signed)]

    return [BaseTag(tag) for tag in signed]


def _find_scope_path(root: "Dataset", target: "Dataset") -> _ScopePath | None:
    """Return the path from ``root`` to ``target`` through sequence items.

    Uses object identity; returns ``None`` if ``target`` is not reachable
    from ``root``.
    """
    if root is target:
        return ()

    for elem in root:
        if elem.VR != "SQ" or elem.value is None:
            continue
        for index, item in enumerate(elem.value):
            sub_path = _find_scope_path(item, target)
            if sub_path is not None:
                return ((elem.tag, index),) + sub_path

    return None


class SignatureEditPlan:
    """Plan signature-aware edits to a :class:`~pydicom.dataset.Dataset`.

    Pending changes are registered with :meth:`change` without modifying the
    dataset. :attr:`invalidated_signatures` reports which digital signatures
    (at any nesting level) would be invalidated by those changes, based on
    the *Data Elements Signed* coverage in each signature's
    *MAC Parameters Sequence*. :meth:`commit` then either rejects the edit
    (``on_invalidated='raise'``, the default) or removes the affected
    signatures (``on_invalidated='remove'``) before applying the changes.

    When removing signatures, *MAC Parameters Sequence* items kept at the
    dataset level are only discarded if their *MAC ID Number* is no longer
    referenced by any remaining signature, so no dangling references are
    left behind.

    Parameters
    ----------
    dataset : Dataset
        The root dataset the plan applies to. Changes to nested sequence
        item datasets are supported as long as they are reachable from this
        dataset.
    """

    def __init__(self, dataset: "Dataset") -> None:
        self._root = dataset
        # (scope dataset, original key, resolved tag, new value, scope path)
        self._changes: list[tuple[Dataset, Any, BaseTag, Any, _ScopePath]] = []

    @property
    def dataset(self) -> "Dataset":
        """Return the root dataset of the plan."""
        return self._root

    @property
    def pending(self) -> bool:
        """Return ``True`` if the plan has uncommitted changes."""
        return bool(self._changes)

    def change(self, *args: Any) -> None:
        """Register a pending element change.

        May be called as ``change(key, value)`` to change an element of the
        plan's root dataset, or as ``change(dataset, key, value)`` to change
        an element of a nested sequence item dataset. ``key`` may be an
        element keyword or tag.

        Changing an element inside a sequence item also marks the enclosing
        sequence element (and any ancestors) as changed, so signatures
        covering the sequence itself are invalidated as well.
        """
        if len(args) == 2:
            scope, key, value = self._root, args[0], args[1]
        elif len(args) == 3:
            scope, key, value = args
        else:
            raise TypeError("change() takes (key, value) or (dataset, key, value)")

        path = _find_scope_path(self._root, scope)
        if path is None:
            raise ValueError(
                "The given dataset is not reachable from the plan's root dataset"
            )

        self._changes.append((scope, key, _resolve_tag(key), value, path))

    def _dirty_tags(
        self,
    ) -> dict[int, tuple["Dataset", set[BaseTag], _ScopePath]]:
        """Return changed tags per scope, propagating nested changes upward."""
        dirty: dict[int, tuple[Dataset, set[BaseTag], _ScopePath]] = {}

        def mark(scope: "Dataset", tag: BaseTag, path: _ScopePath) -> None:
            entry = dirty.setdefault(id(scope), (scope, set(), path))
            entry[1].add(tag)

        for scope, _key, tag, _value, path in self._changes:
            mark(scope, tag, path)
            # A change inside a sequence item changes the sequence element
            # itself, all the way up to the root dataset
            ancestor = self._root
            ancestor_path: _ScopePath = ()
            for seq_tag, index in path:
                mark(ancestor, seq_tag, ancestor_path)
                ancestor = ancestor[seq_tag].value[index]
                ancestor_path = ancestor_path + ((seq_tag, index),)

        return dirty

    @property
    def invalidated_signatures(self) -> list[InvalidatedSignature]:
        """Return the signatures that the pending changes would invalidate.

        Only signatures whose *Data Elements Signed* coverage intersects the
        pending changes are included; signatures covering unrelated elements
        are unaffected.
        """
        invalidated: list[InvalidatedSignature] = []
        for scope, tags, path in self._dirty_tags().values():
            signatures = scope.get("DigitalSignaturesSequence")
            if signatures is None:
                continue
            for index, item in enumerate(signatures):
                signed = _signed_tags(item)
                hits = sorted(set(signed) & tags)
                if not hits:
                    continue
                mac_seq = item.get("MACParametersSequence")
                mac_id = None
                if mac_seq is not None and len(mac_seq):
                    mac_id = mac_seq[0].get("MACIDNumber")
                invalidated.append(
                    InvalidatedSignature(
                        scope=scope,
                        index=index,
                        signature_uid=item.get("DigitalSignatureUID"),
                        mac_id=mac_id,
                        data_elements_signed=signed,
                        triggered_by=hits,
                        path=path,
                    )
                )

        return invalidated

    def commit(
        self, on_invalidated: Literal["raise", "remove"] = "raise"
    ) -> list[InvalidatedSignature]:
        """Apply the pending changes.

        Parameters
        ----------
        on_invalidated : {"raise", "remove"}, optional
            If ``"raise"`` (default) and any signature would be invalidated,
            raise :class:`SignatureInvalidationError` without modifying the
            dataset. If ``"remove"``, remove the invalidated signatures (and
            any *MAC Parameters Sequence* items no longer referenced by the
            remaining signatures) before applying the changes.

        Returns
        -------
        list of InvalidatedSignature
            The signatures invalidated by the applied changes (empty if
            none).
        """
        if on_invalidated not in ("raise", "remove"):
            raise ValueError(
                f"Invalid on_invalidated value {on_invalidated!r}, "
                "must be 'raise' or 'remove'"
            )

        invalidated = self.invalidated_signatures
        if invalidated:
            if on_invalidated == "raise":
                raise SignatureInvalidationError(invalidated)
            self._remove_signatures(invalidated)

        for scope, key, _tag, value, _path in self._changes:
            _apply_change(scope, key, value)

        self._changes.clear()
        return invalidated

    def discard(self) -> None:
        """Discard all pending changes without modifying the dataset."""
        self._changes.clear()

    def _remove_signatures(self, invalidated: list[InvalidatedSignature]) -> None:
        """Remove invalidated signature items and unreferenced MAC parameters."""
        by_scope: dict[int, tuple[Dataset, list[int]]] = {}
        for sig in invalidated:
            entry = by_scope.setdefault(id(sig.scope), (sig.scope, []))
            entry[1].append(sig.index)

        for scope, indices in by_scope.values():
            signatures = scope.get("DigitalSignaturesSequence")
            for index in sorted(set(indices), reverse=True):
                del signatures[index]
            if not signatures:
                del scope[DIGITAL_SIGNATURES_SEQUENCE]

            self._prune_mac_parameters(scope)

    @staticmethod
    def _prune_mac_parameters(scope: "Dataset") -> None:
        """Remove dataset-level MAC parameters no signature references anymore.

        *MAC Parameters Sequence* items embedded in remaining signature items
        belong to those signatures and are always kept. Items held directly
        in the scope dataset are shared and referenced by *MAC ID Number*;
        only items still referenced by a remaining signature are kept.
        """
        shared = scope.get("MACParametersSequence")
        if shared is None:
            return

        referenced: set[Any] = set()
        signatures = scope.get("DigitalSignaturesSequence")
        if signatures is not None:
            for item in signatures:
                mac_seq = item.get("MACParametersSequence")
                if mac_seq is None:
                    continue
                for mac in mac_seq:
                    mac_id = mac.get("MACIDNumber")
                    if mac_id is not None:
                        referenced.add(mac_id)

        for index in range(len(shared) - 1, -1, -1):
            if shared[index].get("MACIDNumber") not in referenced:
                del shared[index]

        if not shared:
            del scope[MAC_PARAMETERS_SEQUENCE]
