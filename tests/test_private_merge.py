# Copyright 2008-2025 pydicom authors. See LICENSE file for details.
"""Tests for Dataset.merge_private_tags()."""

import pytest

from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset
from pydicom.filebase import DicomBytesIO


def add_block(ds, group, creator, elements, slot=0x10):
    """Add a private creator at 'slot' and its data 'elements'.

    'elements' is a list of (element_offset, VR, value) tuples.
    """
    ds.add_new((group, slot), "LO", creator)
    for offset, vr, value in elements:
        ds.add_new((group, (slot << 8) | offset), vr, value)


def roundtrip(ds):
    """Write 'ds' to an in-memory buffer and read it back."""
    buffer = DicomBytesIO()
    dcmwrite(buffer, ds, implicit_vr=False, little_endian=True)
    buffer.seek(0)
    return dcmread(buffer, force=True)


class TestMergePrivateTags:
    def test_slot_swap_by_creator_identity(self):
        """Same numeric slot, different creators: no numeric overwrite."""
        target = Dataset()
        add_block(target, 0x0009, "TARGET", [(0x01, "LO", "target value")])
        source = Dataset()
        add_block(
            source,
            0x0009,
            "SOURCE",
            [(0x01, "LO", "source value"), (0x02, "UN", b"\x01\x02\x03")],
        )

        mapping = target.merge_private_tags(source)

        # target block at 0x10 untouched, source block allocated at 0x11
        assert target[0x00090010].value == "TARGET"
        assert target[0x00091001].value == "target value"
        assert target[0x00090011].value == "SOURCE"
        assert target[0x00091101].value == "source value"
        # unknown VR and raw value preserved
        assert target[0x00091102].VR == "UN"
        assert target[0x00091102].value == b"\x01\x02\x03"

        assert mapping[(0x00090010,)] == 0x00090011
        assert mapping[(0x00091001,)] == 0x00091101
        assert mapping[(0x00091002,)] == 0x00091102

    def test_matching_creator_same_slot(self):
        """Same creator in both datasets merges into the existing block."""
        target = Dataset()
        add_block(target, 0x0009, "CREATOR", [(0x01, "LO", "target")])
        source = Dataset()
        add_block(source, 0x0009, "CREATOR", [(0x02, "LO", "source")])

        mapping = target.merge_private_tags(source)

        assert target[0x00091001].value == "target"
        assert target[0x00091002].value == "source"
        assert 0x00090011 not in target
        assert mapping[(0x00091002,)] == 0x00091002

    def test_creator_with_multiple_blocks(self):
        """A creator occupying several blocks keeps its element relations."""
        target = Dataset()
        add_block(target, 0x0009, "MULTI", [(0x01, "LO", "target")])
        source = Dataset()
        add_block(source, 0x0009, "MULTI", [(0x01, "LO", "ignored")], slot=0x10)
        add_block(source, 0x0009, "MULTI", [(0x02, "LO", "second")], slot=0x20)

        mapping = target.merge_private_tags(source, on_conflict="skip")

        # first block matched the existing creator block, conflict skipped
        assert target[0x00091001].value == "target"
        # second block allocated a new slot with the same creator
        assert target[0x00090011].value == "MULTI"
        assert target[0x00091102].value == "second"
        assert mapping[(0x00092002,)] == 0x00091102

    def test_conflict_policies(self):
        """Conflicts in an existing creator block follow 'on_conflict'."""
        for policy, expected in (
            ("skip", "target"),
            ("overwrite", "source"),
        ):
            target = Dataset()
            add_block(target, 0x0009, "CREATOR", [(0x01, "LO", "target")])
            source = Dataset()
            add_block(source, 0x0009, "CREATOR", [(0x01, "LO", "source")])

            target.merge_private_tags(source, on_conflict=policy)
            assert target[0x00091001].value == expected

        target = Dataset()
        add_block(target, 0x0009, "CREATOR", [(0x01, "LO", "target")])
        source = Dataset()
        add_block(source, 0x0009, "CREATOR", [(0x01, "LO", "source")])
        with pytest.raises(ValueError, match="Merge conflict"):
            target.merge_private_tags(source)
        # failed merge leaves the target unchanged
        assert target[0x00091001].value == "target"
        assert len(target) == 2

    def test_invalid_on_conflict(self):
        with pytest.raises(ValueError, match="Invalid on_conflict"):
            Dataset().merge_private_tags(Dataset(), on_conflict="bogus")

    def test_nested_items_have_own_allocation_space(self):
        """Each sequence item gets its own private creator slot space."""
        target = Dataset()
        item0 = Dataset()
        add_block(item0, 0x0009, "CREATOR A", [(0x01, "LO", "a")])
        target.add_new((0x0040, 0xA730), "SQ", [item0, Dataset()])

        source = Dataset()
        src0 = Dataset()
        add_block(src0, 0x0009, "CREATOR B", [(0x01, "LO", "b0")])
        src1 = Dataset()
        add_block(src1, 0x0009, "CREATOR B", [(0x01, "LO", "b1")])
        source.add_new((0x0040, 0xA730), "SQ", [src0, src1])

        mapping = target.merge_private_tags(source)

        items = target[0x0040A730].value
        # item 0: 0x10 occupied by CREATOR A, so CREATOR B goes to 0x11
        assert items[0][0x00091101].value == "b0"
        # item 1: own allocation space, CREATOR B gets slot 0x10
        assert items[1][0x00091001].value == "b1"

        assert (
            mapping[
                (
                    0x0040A730,
                    0,
                    0x00091001,
                )
            ]
            == 0x00091101
        )
        assert (
            mapping[
                (
                    0x0040A730,
                    1,
                    0x00091001,
                )
            ]
            == 0x00091001
        )

    def test_source_items_appended(self):
        """Source sequence items without target counterpart are appended."""
        target = Dataset()
        target.add_new((0x0040, 0xA730), "SQ", [Dataset()])
        source = Dataset()
        src0 = Dataset()
        add_block(src0, 0x0009, "CREATOR", [(0x01, "LO", "first")])
        src1 = Dataset()
        add_block(src1, 0x0009, "CREATOR", [(0x01, "LO", "appended")])
        source.add_new((0x0040, 0xA730), "SQ", [src0, src1])

        mapping = target.merge_private_tags(source)

        items = target[0x0040A730].value
        assert len(items) == 2
        assert items[0][0x00091001].value == "first"
        assert items[1][0x00091001].value == "appended"
        assert (
            mapping[
                (
                    0x0040A730,
                    1,
                    0x00091001,
                )
            ]
            == 0x00091001
        )

    def test_private_sequence_copied_wholesale(self):
        """A private SQ is copied with its items; inner tags are mapped."""
        target = Dataset()
        source = Dataset()
        item = Dataset()
        add_block(item, 0x0011, "INNER", [(0x05, "LO", "inner")])
        source.add_new((0x0009, 0x0010), "LO", "OUTER")
        source.add_new((0x0009, 0x1001), "SQ", [item])

        mapping = target.merge_private_tags(source)

        copied = target[0x00091001].value[0]
        assert copied[0x00110010].value == "INNER"
        assert copied[0x00111005].value == "inner"
        assert mapping[(0x00091001,)] == 0x00091001
        assert (
            mapping[
                (
                    0x00091001,
                    0,
                    0x00111005,
                )
            ]
            == 0x00111005
        )

    def test_missing_creator_raises_before_modification(self):
        target = Dataset()
        source = Dataset()
        source.add_new((0x0009, 0x1001), "LO", "orphan")

        with pytest.raises(ValueError, match="no.*private creator"):
            target.merge_private_tags(source)
        assert len(target) == 0

    def test_empty_creator_raises(self):
        target = Dataset()
        source = Dataset()
        add_block(source, 0x0009, "  ", [(0x01, "LO", "value")])

        with pytest.raises(ValueError, match="has no value"):
            target.merge_private_tags(source)
        assert len(target) == 0

    @pytest.mark.parametrize("element", [0x0005, 0x0101, 0x0FFF])
    def test_illegal_private_tag_raises(self, element):
        target = Dataset()
        source = Dataset()
        source.add_new((0x0009, element), "LO", "illegal")

        with pytest.raises(ValueError, match="Illegal private tag"):
            target.merge_private_tags(source)
        assert len(target) == 0

    def test_slot_exhaustion_raises(self):
        target = Dataset()
        for slot in range(0x10, 0x100):
            target.add_new((0x0009, slot), "LO", f"CREATOR {slot:02X}")
        source = Dataset()
        add_block(source, 0x0009, "NEW CREATOR", [(0x01, "LO", "value")])

        with pytest.raises(ValueError, match="No free private creator slot"):
            target.merge_private_tags(source)
        assert len(target) == 0xF0

    def test_write_read_roundtrip(self):
        """After write/read, blocks are found by creator and offset."""
        target = Dataset()
        add_block(target, 0x0009, "TARGET", [(0x01, "LO", "target value")])
        item = Dataset()
        add_block(item, 0x0011, "NESTED", [(0x03, "UN", b"\xaa\xbb")])
        target.add_new((0x0040, 0xA730), "SQ", [item, Dataset()])

        source = Dataset()
        add_block(
            source,
            0x0009,
            "SOURCE",
            [(0x01, "LO", "source value"), (0x02, "UN", b"\x01\x02")],
        )
        src_item = Dataset()
        add_block(src_item, 0x0011, "NESTED", [(0x04, "LO", "nested source")])
        src_item2 = Dataset()
        add_block(src_item2, 0x0011, "NESTED", [(0x04, "LO", "second item")])
        source.add_new((0x0040, 0xA730), "SQ", [src_item, src_item2])

        target.merge_private_tags(source)
        result = roundtrip(target)

        # every merged private block is retrievable by creator and offset
        assert result.private_block(0x0009, "TARGET")[0x01].value == ("target value")
        block = result.private_block(0x0009, "SOURCE")
        assert block[0x01].value == "source value"
        assert block[0x02].VR == "UN"
        assert block[0x02].value == b"\x01\x02"

        items = result[0x0040A730].value
        nested = items[0].private_block(0x0011, "NESTED")
        assert nested[0x03].value == b"\xaa\xbb"
        assert nested[0x04].value == "nested source"
        assert items[1].private_block(0x0011, "NESTED")[0x04].value == ("second item")
