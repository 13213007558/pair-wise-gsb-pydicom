# Copyright 2008-2026 pydicom authors. See LICENSE file for details.
"""Merge private data elements between datasets by private creator identity.

Private data elements are addressed by the *private creator* element that
reserves a block of 256 element offsets within an odd (private) group, not
by their numeric tag alone. Two datasets may therefore use the same numeric
tag for completely different private elements, or the same creator at
different block slots. This module provides :func:merge_private_elements,
which merges the private elements of a source dataset into a target dataset
by matching or allocating block slots per private creator instead of
overwriting numeric tags.
"""

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydicom.tag import BaseTag, Tag

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataelem import DataElement
    from pydicom.dataset import Dataset


#: Valid values for the "conflict" parameter of :func:merge_private_elements
CONFLICT_POLICIES = ("error", "skip", "overwrite")

#: Issue kind for a private data element without a matching private creator
MISSING_CREATOR = "missing_creator"
#: Issue kind for a private element in the reserved element range
INVALID_PRIVATE_TAG = "invalid_private_tag"
#: Issue kind when no free private creator slot is available in the target
SLOT_EXHAUSTED = "slot_exhausted"
#: Issue kind for conflicting elements with conflict="error"
CONFLICT = "conflict"

#: A path to an element within a (possibly nested) dataset, alternating
#: between tag (as int) and sequence item index, e.g. (0x00091001,)
#: or (0x00081115, 0, 0x00091001)
TagPath = tuple[int, ...]

ConflictPolicy = Literal["error", "skip", "overwrite"]


@dataclass
class PrivateMergeIssue:
    """A problem found while planning a private element merge.

    Attributes
    ----------
    kind : str
        One of "missing_creator", "invalid_private_tag",
        "slot_exhausted" or "conflict".
    path : tuple of int
        The path to the offending element in the source dataset, alternating
        between tag and sequence item index (see :data:TagPath).
    message : str
        A human readable description of the problem.
    """

    kind: str
    path: TagPath
    message: str


class PrivateMergeError(ValueError):
    """Raised when private elements cannot be merged.

    The target dataset is guaranteed to be unmodified when this exception
    is raised. The individual problems are available in
    :attr:PrivateMergeError.issues.
    """

    def __init__(self, issues: list[PrivateMergeIssue]) -> None:
        self.issues = list(issues)
        details = "\n".join(f"  - {issue.message}" for issue in self.issues)
        super().__init__(f"Cannot merge private elements:\n{details}")


def _format_path(path: TagPath) -> str:
    """Return a readable representation of a tag path."""
    parts = []
    for index, component in enumerate(path):
        if index % 2 == 0:
            tag = BaseTag(component)
            parts.append(f"({tag.group:04X},{tag.element:04X})")
        else:
            parts.append(f"[{component}]")
    return "".join(parts)


@dataclass
class _MergePlan:
    """Accumulates issues, actions and the tag mapping during planning."""

    issues: list[PrivateMergeIssue] = field(default_factory=list)
    # ("creator", target_ds, group, slot_element, creator) or
    # ("element", target_ds, new_tag, source_element)
    actions: list[tuple] = field(default_factory=list)
    mapping: dict[TagPath, TagPath] = field(default_factory=dict)
    # id(dataset) -> set of reserved (group, slot element) per dataset level
    reserved: dict[int, set[tuple[int, int]]] = field(default_factory=dict)


def _target_slot_for(
    plan: _MergePlan,
    target: "Dataset",
    group: int,
    creator: Any,
    source_slot: int,
    old_prefix: TagPath,
    new_prefix: TagPath,
) -> int | None:
    """Return the target slot element for the given group and creator.

    Uses the existing block if the target already contains the creator in
    the group, otherwise reserves the first free slot. Returns None and
    records an issue if no slot is available.
    """
    existing = {
        elem.tag.element
        for elem in target
        if elem.tag.group == group and elem.tag.is_private_creator
    }
    slot = next(
        (
            elem.tag.element
            for elem in target
            if elem.tag.group == group
            and elem.tag.is_private_creator
            and elem.value == creator
        ),
        None,
    )
    if slot is None:
        reserved = plan.reserved.setdefault(id(target), set())
        slot = next(
            (
                candidate
                for candidate in range(0x10, 0x100)
                if candidate not in existing and (group, candidate) not in reserved
            ),
            None,
        )
        if slot is None:
            plan.issues.append(
                PrivateMergeIssue(
                    SLOT_EXHAUSTED,
                    old_prefix + ((group << 16) | source_slot,),
                    f"No free private creator slot in group 0x{group:04X} of the "
                    f"target dataset for creator {creator!r}",
                )
            )
            return None
        reserved.add((group, slot))
        plan.actions.append(("creator", target, group, slot, creator))

    source_creator_tag = (group << 16) | source_slot
    target_creator_tag = (group << 16) | slot
    old_path = old_prefix + (source_creator_tag,)
    if old_path not in plan.mapping:
        plan.mapping[old_path] = new_prefix + (target_creator_tag,)
    return slot


def _plan_copied_sq(
    plan: _MergePlan, source_elem: "DataElement", old_path: TagPath, new_path: TagPath
) -> None:
    """Record mapping entries for private content of a copied SQ element.

    A copied sequence keeps its internal structure, so the nested tags are
    unchanged and only the path prefix differs between source and target.
    """

    def walk(source_ds: "Dataset", old_prefix: TagPath, new_prefix: TagPath) -> None:
        for elem in source_ds:
            tag = int(elem.tag)
            if elem.VR == "SQ":
                for index, item in enumerate(elem.value):
                    walk(item, old_prefix + (tag, index), new_prefix + (tag, index))
            elif elem.tag.is_private:
                plan.mapping[old_prefix + (tag,)] = new_prefix + (tag,)

    for index, item in enumerate(source_elem.value):
        walk(item, old_path + (index,), new_path + (index,))


def _has_creator(creators: dict[tuple[int, int], Any], key: tuple[int, int]) -> bool:
    """Return True if a usable private creator exists at the given slot."""
    creator = creators.get(key)
    if creator is None:
        return False
    if isinstance(creator, str) and not creator.strip():
        return False
    return True


def _plan_level(
    plan: _MergePlan,
    target: "Dataset",
    source: "Dataset",
    old_prefix: TagPath,
    new_prefix: TagPath,
    conflict: ConflictPolicy,
) -> None:
    """Plan the merge of private elements from source into target.

    Only reads both datasets; all modifications are deferred to the actions
    recorded in the plan. Each dataset level (top-level dataset or sequence
    item) has its own private creator slot allocation space.
    """
    creators: dict[tuple[int, int], Any] = {}
    for elem in source:
        if elem.tag.is_private_creator:
            creators[(elem.tag.group, elem.tag.element)] = elem.value

    block_slots: dict[tuple[int, Any], int | None] = {}

    def slot_for(group: int, source_slot: int) -> int | None:
        creator = creators[(group, source_slot)]
        key = (group, creator)
        if key not in block_slots:
            block_slots[key] = _target_slot_for(
                plan, target, group, creator, source_slot, old_prefix, new_prefix
            )
        return block_slots[key]

    for elem in source:
        tag = elem.tag
        if not tag.is_private:
            # recurse into sequences present in both datasets; every item
            # gets its own slot allocation space
            if elem.VR == "SQ" and tag in target and target[tag].VR == "SQ":
                for index, (target_item, source_item) in enumerate(
                    zip(target[tag].value, elem.value)
                ):
                    _plan_level(
                        plan,
                        target_item,
                        source_item,
                        old_prefix + (int(tag), index),
                        new_prefix + (int(tag), index),
                        conflict,
                    )
            continue

        if tag.is_private_creator or tag.element == 0x0000:
            continue

        if tag.element < 0x1000:
            plan.issues.append(
                PrivateMergeIssue(
                    INVALID_PRIVATE_TAG,
                    old_prefix + (int(tag),),
                    f"Invalid private tag {_format_path(old_prefix + (int(tag),))}: "
                    f"element 0x{tag.element:04X} lies in the reserved range "
                    f"0x0001-0x0FFF",
                )
            )
            continue

        source_slot = tag.element >> 8
        if not _has_creator(creators, (tag.group, source_slot)):
            plan.issues.append(
                PrivateMergeIssue(
                    MISSING_CREATOR,
                    old_prefix + (int(tag),),
                    f"No private creator found for element "
                    f"{_format_path(old_prefix + (int(tag),))} "
                    f"(expected at slot 0x{source_slot:02X})",
                )
            )
            continue

        target_slot = slot_for(tag.group, source_slot)
        if target_slot is None:
            continue

        new_tag = (tag.group << 16) | (target_slot << 8) | (tag.element & 0xFF)
        old_path = old_prefix + (int(tag),)
        new_path = new_prefix + (new_tag,)
        plan.mapping[old_path] = new_path

        if BaseTag(new_tag) in target:
            if conflict == "error":
                plan.issues.append(
                    PrivateMergeIssue(
                        CONFLICT,
                        old_path,
                        f"Element {_format_path(old_path)} conflicts with "
                        f"existing target element {_format_path(new_path)}",
                    )
                )
                continue
            if conflict == "skip":
                continue

        plan.actions.append(("element", target, new_tag, elem))
        if elem.VR == "SQ":
            _plan_copied_sq(plan, elem, old_path, new_path)


def _apply(plan: _MergePlan) -> None:
    """Execute the recorded merge actions."""
    for action in plan.actions:
        if action[0] == "creator":
            _, target, group, slot, creator = action
            target.add_new(Tag(group, slot), "LO", creator)
        else:
            _, target, new_tag, elem = action
            new_elem = copy.deepcopy(elem)
            new_elem.tag = Tag(new_tag)
            target.add(new_elem)


def merge_private_elements(
    target: "Dataset",
    source: "Dataset",
    *,
    conflict: ConflictPolicy = "error",
) -> dict[TagPath, TagPath]:
    """Merge the private elements of source into target by private
    creator identity.

    Private elements are matched or allocated by their private creator
    string within each odd group, never by numeric tag alone. The element
    offset within a creator's block is preserved, so elements of a creator
    that occupies several blocks (or several groups) keep their relative
    positions. Sequences present in both datasets are recursed into, and
    every sequence item gets its own slot allocation space. Elements are
    copied with their VR and value unchanged, including unknown (UN)
    VRs and their raw byte values.

    The merge is planned completely before anything is modified: if any
    problem is found (missing private creator, exhausted creator slots,
    invalid private tags, or conflicts with conflict="error"),
    :exc:PrivateMergeError is raised and target is left unmodified.

    Parameters
    ----------
    target : Dataset
        The dataset to merge into. Modified in place.
    source : Dataset
        The dataset to take private elements from. Not modified.
    conflict : str, optional
        How to handle a private element whose target position is already
        occupied: "error" (default) reports the conflict and aborts,
        "skip" keeps the target element, "overwrite" replaces the
        target element with the source element.

    Returns
    -------
    dict of tuple of int to tuple of int
        A mapping from the old (source) tag path to the new (target) tag
        path for every merged private element, including private creator
        elements. A path alternates between tag (as int) and sequence
        item index, e.g. (0x00091001,) for a top-level element or
        (0x00081115, 0, 0x00091001) for an element in the first item
        of a sequence. This allows updating external references that point
        to private elements.

    Raises
    ------
    PrivateMergeError
        If the merge cannot be planned without problems. The individual
        issues are available in PrivateMergeError.issues; target
        is not modified in this case.
    ValueError
        If conflict is not one of "error", "skip" or "overwrite".
    """
    if conflict not in CONFLICT_POLICIES:
        raise ValueError(
            f"Unknown conflict policy '{conflict}', must be one of "
            f"{CONFLICT_POLICIES}"
        )

    plan = _MergePlan()
    _plan_level(plan, target, source, (), (), conflict)
    if plan.issues:
        raise PrivateMergeError(plan.issues)
    _apply(plan)
    return plan.mapping
