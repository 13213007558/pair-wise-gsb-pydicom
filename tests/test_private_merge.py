# Copyright 2008-2026 pydicom authors. See LICENSE file for details.
"""Test suite for pydicom.util.private_merge."""

import pytest

from pydicom import dcmread, Dataset
from pydicom.dataelem import DataElement
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.util.private_merge import (
    CONFLICT,
    INVALID_PRIVATE_TAG,
    MISSING_CREATOR,
    SLOT_EXHAUSTED,
    PrivateMergeError,
    merge_private_elements,
)

SQ_TAG = 0x00081115  # Referenced Series Sequence


def add_private(ds, group, creator, offset, vr, value):
    """Add a private element via its creator block and return its tag."""
    block = ds.private_block(group, creator, create=True)
    block.add_new(offset, vr, value)
    return int(block.get_tag(offset))


def make_conflict_pair():
    """Return target/source with the same creator, slot and offset."""
    target = Dataset()
    add_private(target, 0x0009, "CREATOR A", 0x01, "LO", "old")
    source = Dataset()
    add_private(source, 0x0009, "CREATOR A", 0x01, "LO", "new")
    return target, source


class TestMergePrivateElements:
    def test_new_creator_allocated_free_slot(self):
        """A new creator is allocated the first free slot, no overwriting."""
        target = Dataset()
        add_private(target, 0x0009, "CREATOR A", 0x01, "LO", "target")
        source = Dataset()
        add_private(source, 0x0009, "CREATOR B", 0x01, "LO", "source")

        mapping = merge_private_elements(target, source)

        assert target[0x00091001].value == "target"
        assert target[0x00091101].value == "source"
        assert target[0x00090011].value == "CREATOR B"
        assert mapping[(0x00091001,)] == (0x00091101,)
        # the private creator element itself is mapped as well
        assert mapping[(0x00090010,)] == (0x00090011,)

    def test_slot_swap_same_creator(self):
        """The same creator at different slots is matched by identity."""
        target = Dataset()
        add_private(target, 0x0009, "OTHER", 0x01, "LO", "other")
        add_private(target, 0x0009, "CREATOR A", 0x01, "LO", "old")  # slot 0x11
        source = Dataset()
        add_private(source, 0x0009, "CREATOR A", 0x05, "LO", "swapped")  # slot 0x10

        mapping = merge_private_elements(target, source)

        # merged into the existing block of CREATOR A at slot 0x11
        assert target[0x00091105].value == "swapped"
        assert 0x00091005 not in target
        creators = [elem for elem in target if elem.tag.is_private_creator]
        assert len(creators) == 2
        assert mapping[(0x00091005,)] == (0x00091105,)
        assert mapping[(0x00090010,)] == (0x00090011,)

    def test_creator_in_multiple_blocks(self):
        """A creator occupying several blocks keeps its element relations."""
        source = Dataset()
        source.add_new(0x00090010, "LO", "MULTI")
        source.add_new(0x00090011, "LO", "MULTI")
        source.add_new(0x00091001, "LO", "first")
        source.add_new(0x00091102, "LO", "second")
        source.add_new(0x00110010, "LO", "MULTI")
        source.add_new(0x00111007, "LO", "other group")
        target = Dataset()

        mapping = merge_private_elements(target, source)

        # both blocks of MULTI in group 0x0009 merge into a single target
        # block, element offsets are preserved
        assert target[0x00091001].value == "first"
        assert target[0x00091002].value == "second"
        assert target[0x00111007].value == "other group"
        assert target[0x00090010].value == "MULTI"
        assert 0x00090011 not in target
        assert mapping[(0x00091102,)] == (0x00091002,)
        assert mapping[(0x00111007,)] == (0x00111007,)

    def test_conflict_error(self):
        """conflict='error' reports the conflict and modifies nothing."""
        target, source = make_conflict_pair()
        with pytest.raises(PrivateMergeError) as exc:
            merge_private_elements(target, source)
        assert exc.value.issues[0].kind == CONFLICT
        assert exc.value.issues[0].path == (0x00091001,)
        assert target[0x00091001].value == "old"

    def test_conflict_skip(self):
        """conflict='skip' keeps the target element."""
        target, source = make_conflict_pair()
        mapping = merge_private_elements(target, source, conflict="skip")
        assert target[0x00091001].value == "old"
        assert mapping[(0x00091001,)] == (0x00091001,)

    def test_conflict_overwrite(self):
        """conflict='overwrite' replaces the target element."""
        target, source = make_conflict_pair()
        merge_private_elements(target, source, conflict="overwrite")
        assert target[0x00091001].value == "new"

    def test_invalid_conflict_policy(self):
        target, source = make_conflict_pair()
        with pytest.raises(ValueError, match="Unknown conflict policy"):
            merge_private_elements(target, source, conflict="replace")

    def test_missing_creator_reported_before_modify(self):
        """A private element without creator aborts the merge."""
        source = Dataset()
        source.add_new(0x00091001, "LO", "orphan")  # no creator at slot 0x10
        add_private(source, 0x0011, "CREATOR B", 0x01, "LO", "valid")
        target = Dataset()

        with pytest.raises(PrivateMergeError) as exc:
            merge_private_elements(target, source)

        assert [issue.kind for issue in exc.value.issues] == [MISSING_CREATOR]
        assert exc.value.issues[0].path == (0x00091001,)
        # the valid creator was not applied either
        assert len(target) == 0

    def test_empty_creator_counts_as_missing(self):
        source = Dataset()
        source.add_new(0x00090010, "LO", "")
        source.add_new(0x00091001, "LO", "orphan")
        target = Dataset()

        with pytest.raises(PrivateMergeError) as exc:
            merge_private_elements(target, source)

        assert [issue.kind for issue in exc.value.issues] == [MISSING_CREATOR]
        assert len(target) == 0

    def test_slot_exhausted_reported_before_modify(self):
        """No free creator slot in the target group aborts the merge."""
        target = Dataset()
        for slot in range(0x10, 0x100):
            target.add_new((0x0009, slot), "LO", f"CREATOR {slot:02X}")
        source = Dataset()
        add_private(source, 0x0009, "NEWCOMER", 0x01, "LO", "x")

        with pytest.raises(PrivateMergeError) as exc:
            merge_private_elements(target, source)

        assert [issue.kind for issue in exc.value.issues] == [SLOT_EXHAUSTED]
        assert len(target) == 0x100 - 0x10
        assert 0x00091001 not in target

    def test_invalid_private_tag_reported_before_modify(self):
        """A private tag in the reserved element range aborts the merge."""
        source = Dataset()
        source[0x00090501] = DataElement(0x00090501, "UN", b"\x00\x01")
        target = Dataset()

        with pytest.raises(PrivateMergeError) as exc:
            merge_private_elements(target, source)

        assert [issue.kind for issue in exc.value.issues] == [INVALID_PRIVATE_TAG]
        assert exc.value.issues[0].path == (0x00090501,)
        assert len(target) == 0

    def test_unknown_vr_and_raw_value_preserved(self):
        """UN elements keep their VR and raw byte value."""
        source = Dataset()
        add_private(source, 0x0009, "RAW", 0x01, "UN", b"\x00\x01\xfe\xff")
        target = Dataset()

        merge_private_elements(target, source)

        elem = target[0x00091001]
        assert elem.VR == "UN"
        assert elem.value == b"\x00\x01\xfe\xff"

    def test_lone_creator_not_merged(self):
        """A creator element without data elements is not merged."""
        source = Dataset()
        source.private_block(0x0009, "EMPTY BLOCK", create=True)
        target = Dataset()

        mapping = merge_private_elements(target, source)

        assert mapping == {}
        assert len(target) == 0

    def test_nested_sq_items_independent_allocation(self):
        """Each sequence item gets its own slot allocation space."""
        target_item0 = Dataset()
        add_private(target_item0, 0x0009, "NESTED", 0x01, "LO", "t0")
        target_item1 = Dataset()
        add_private(target_item1, 0x0009, "NESTED", 0x01, "LO", "t1")
        add_private(target_item1, 0x0009, "OTHER", 0x01, "LO", "o1")
        target = Dataset()
        target.add_new(SQ_TAG, "SQ", [target_item0, target_item1])

        source_item0 = Dataset()
        add_private(source_item0, 0x0009, "NESTED", 0x02, "LO", "s0")
        add_private(source_item0, 0x0009, "ZERO", 0x03, "LO", "z0")
        source_item1 = Dataset()
        add_private(source_item1, 0x0009, "NESTED", 0x01, "LO", "s1")
        add_private(source_item1, 0x0009, "NEW", 0x03, "LO", "n1")
        source = Dataset()
        source.add_new(SQ_TAG, "SQ", [source_item0, source_item1])

        mapping = merge_private_elements(target, source, conflict="overwrite")

        t_item0, t_item1 = target[SQ_TAG].value
        assert t_item0[0x00091002].value == "s0"
        # ZERO allocated slot 0x11 in item 0
        assert t_item0[0x00091103].value == "z0"
        assert t_item0[0x00090011].value == "ZERO"
        # conflict in item 1 overwritten
        assert t_item1[0x00091001].value == "s1"
        # NEW allocated slot 0x12 in item 1, 0x11 is taken by OTHER
        assert t_item1[0x00091203].value == "n1"
        assert t_item1[0x00090012].value == "NEW"
        # item 0 is not affected by the allocation in item 1
        assert 0x00090012 not in t_item0

        # the mapping contains the sequence item paths
        assert mapping[(SQ_TAG, 0, 0x00091002)] == (SQ_TAG, 0, 0x00091002)
        assert mapping[(SQ_TAG, 0, 0x00091103)] == (SQ_TAG, 0, 0x00091103)
        assert mapping[(SQ_TAG, 1, 0x00091001)] == (SQ_TAG, 1, 0x00091001)
        assert mapping[(SQ_TAG, 1, 0x00091103)] == (SQ_TAG, 1, 0x00091203)
        # creator elements inside items are mapped as well
        assert mapping[(SQ_TAG, 1, 0x00090011)] == (SQ_TAG, 1, 0x00090012)

    def test_private_sq_copied_with_nested_mapping(self):
        """A copied private SQ keeps its content and maps nested paths."""
        source = Dataset()
        block = source.private_block(0x0009, "SQOWNER", create=True)
        item = Dataset()
        add_private(item, 0x0011, "INNER", 0x01, "LO", "inner")
        block.add_new(0x10, "SQ", [item])
        target = Dataset()

        mapping = merge_private_elements(target, source)

        assert target[0x00091010].VR == "SQ"
        copied_item = target[0x00091010].value[0]
        assert copied_item[0x00111001].value == "inner"
        assert copied_item[0x00110010].value == "INNER"
        assert mapping[(0x00091010,)] == (0x00091010,)
        assert mapping[(0x00091010, 0, 0x00111001)] == (0x00091010, 0, 0x00111001)
        assert mapping[(0x00091010, 0, 0x00110010)] == (0x00091010, 0, 0x00110010)

    def test_source_not_modified(self):
        """The source dataset is left unchanged."""
        target, source = make_conflict_pair()
        before = list(source.keys())
        merge_private_elements(target, source, conflict="overwrite")
        assert list(source.keys()) == before
        assert source[0x00091001].value == "new"

    def test_write_read_roundtrip(self, tmp_path):
        """Merged private blocks survive a write/read round trip."""
        target = Dataset()
        add_private(target, 0x0009, "OTHER", 0x01, "LO", "other")
        add_private(target, 0x0009, "CREATOR A", 0x01, "LO", "old")  # slot 0x11
        t_item0 = Dataset()
        add_private(t_item0, 0x0011, "NESTED", 0x01, "LO", "t0")
        t_item1 = Dataset()
        add_private(t_item1, 0x0011, "NESTED", 0x01, "LO", "t1")
        target.add_new(SQ_TAG, "SQ", [t_item0, t_item1])

        source = Dataset()
        add_private(source, 0x0009, "CREATOR A", 0x01, "LO", "new")  # slot 0x10
        add_private(source, 0x0009, "CREATOR A", 0x02, "UN", b"\xfe\xff\x00")
        add_private(source, 0x0011, "CREATOR B", 0x05, "LO", "b-value")
        s_item0 = Dataset()
        add_private(s_item0, 0x0011, "NESTED", 0x02, "LO", "s0")
        s_item1 = Dataset()
        add_private(s_item1, 0x0011, "NESTED", 0x01, "LO", "s1")
        add_private(s_item1, 0x0011, "EXTRA", 0x07, "LO", "extra")
        source.add_new(SQ_TAG, "SQ", [s_item0, s_item1])

        mapping = merge_private_elements(target, source, conflict="overwrite")
        assert mapping[(0x00091001,)] == (0x00091101,)

        target.file_meta = FileMetaDataset()
        target.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        target.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        target.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        path = tmp_path / "merged.dcm"
        target.save_as(path, enforce_file_format=True)
        rt = dcmread(path)

        # slot swap: CREATOR A merged into its existing block at 0x11
        assert rt.get_private_item(0x0009, 0x01, "CREATOR A").value == "new"
        un_elem = rt.get_private_item(0x0009, 0x02, "CREATOR A")
        assert un_elem.VR == "UN"
        assert un_elem.value == b"\xfe\xff\x00"
        assert rt.get_private_item(0x0009, 0x01, "OTHER").value == "other"
        # newly allocated creator
        assert rt.get_private_item(0x0011, 0x05, "CREATOR B").value == "b-value"
        # nested items with the same creator name merged independently
        rt_item0, rt_item1 = rt[SQ_TAG].value
        assert rt_item0.get_private_item(0x0011, 0x01, "NESTED").value == "t0"
        assert rt_item0.get_private_item(0x0011, 0x02, "NESTED").value == "s0"
        assert rt_item1.get_private_item(0x0011, 0x01, "NESTED").value == "s1"
        assert rt_item1.get_private_item(0x0011, 0x07, "EXTRA").value == "extra"
        # every private block is reachable via creator and element offset
        assert rt.private_block(0x0009, "CREATOR A")[0x01].value == "new"
        assert rt.private_block(0x0011, "CREATOR B")[0x05].value == "b-value"
        assert rt_item1.private_block(0x0011, "EXTRA")[0x07].value == "extra"
