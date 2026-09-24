# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Tests for pixels.utils.select_frames and Dataset.select_frames."""

import copy

import pytest

from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import (
    encapsulate,
    encapsulate_extended,
    generate_fragmented_frames,
    generate_frames,
    parse_basic_offsets,
)
from pydicom.filebase import DicomBytesIO
from pydicom.pixels import select_frames
from pydicom.pixels.utils import pack_bits, unpack_bits
from pydicom.uid import (
    EnhancedMRImageStorage,
    ExplicitVRLittleEndian,
    RLELossless,
    generate_uid,
)


INDICES = [4, 1, 4]


def _enhanced_ds(nr_frames=5, rows=2, cols=3, bits_allocated=8):
    """Return a small enhanced multi-frame dataset with ``nr_frames`` frames.

    Frame ``i`` is filled with pixel value ``i`` and the per-frame functional
    groups are marked with the 1-based frame number.
    """
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.SOPClassUID = EnhancedMRImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = str(nr_frames)

    frame_length = rows * cols * (bits_allocated // 8)
    ds.PixelData = bytes(i for i in range(nr_frames) for _ in range(frame_length))

    shared = Dataset()
    measures = Dataset()
    measures.PixelSpacing = [0.5, 0.5]
    shared.PixelMeasuresSequence = [measures]
    ds.SharedFunctionalGroupsSequence = [shared]

    per_frame = []
    for i in range(nr_frames):
        item = Dataset()
        content = Dataset()
        content.FrameAcquisitionNumber = i + 1
        item.FrameContentSequence = [content]
        dimension = Dataset()
        dimension.DimensionIndexValues = [1, i + 1]
        item.DimensionIndexSequence = [dimension]
        per_frame.append(item)

    ds.PerFrameFunctionalGroupsSequence = per_frame

    reference = Dataset()
    reference.ReferencedSOPClassUID = "1.2.3.4"
    reference.ReferencedSOPInstanceUID = generate_uid()
    reference.ReferencedFrameNumber = [1, 3, 5]
    ds.ReferencedImageSequence = [reference]

    return ds


def _roundtrip(ds):
    """Write ``ds`` to an in-memory buffer and read it back."""
    fp = DicomBytesIO()
    ds.save_as(fp, enforce_file_format=True)
    fp.seek(0)
    return dcmread(fp)


def _frame_numbers(ds):
    """Return the per-frame Frame Acquisition Number values."""
    return [
        item.FrameContentSequence[0].FrameAcquisitionNumber
        for item in ds.PerFrameFunctionalGroupsSequence
    ]


class TestSelectFramesNative:
    """Tests for natively encoded (uncompressed) pixel data."""

    def test_roundtrip(self):
        """Pixels, functional groups and references match the selection."""
        ds = _enhanced_ds()
        original = copy.deepcopy(ds)

        out = ds.select_frames(INDICES)

        # The input dataset is unchanged
        assert ds == original

        assert 3 == out.NumberOfFrames
        assert bytes(i for i in INDICES for _ in range(6)) == out.PixelData

        # Per-frame functional groups follow the selection, shared groups
        #   are preserved
        assert [5, 2, 5] == _frame_numbers(out)
        assert [[1, i + 1] for i in INDICES] == [
            item.DimensionIndexSequence[0].DimensionIndexValues[:]
            for item in out.PerFrameFunctionalGroupsSequence
        ]
        assert [0.5, 0.5] == (
            out.SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[:]
        )

        # Repeated frames give independent items
        assert (
            out.PerFrameFunctionalGroupsSequence[0]
            is not out.PerFrameFunctionalGroupsSequence[2]
        )

        # Old frames 1, 3, 5 -> not selected, not selected, new frames 1 + 3
        assert [1, 3] == (out.ReferencedImageSequence[0].ReferencedFrameNumber[:])

        read = _roundtrip(out)
        assert 3 == read.NumberOfFrames
        assert out.PixelData == read.PixelData
        assert [5, 2, 5] == _frame_numbers(read)
        assert [1, 3] == (read.ReferencedImageSequence[0].ReferencedFrameNumber[:])
        assert [0.5, 0.5] == (
            read.SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[:]
        )

    def test_bit_packed(self):
        """Bits Allocated of 1 is repacked across byte boundaries."""
        # 10 pixels per frame, so frame boundaries are not byte aligned
        ds = _enhanced_ds(rows=2, cols=5, bits_allocated=1)
        ds.BitsStored = 1
        ds.HighBit = 0
        # Frame ``i`` is all 0-bits for even ``i`` and all 1-bits for odd ``i``
        bits = bytes(i % 2 for i in range(5) for _ in range(10))
        ds.PixelData = pack_bits(bits)
        original = copy.deepcopy(ds)

        out = select_frames(ds, INDICES)
        assert ds == original

        expected = [i % 2 for i in INDICES for _ in range(10)]
        assert expected == list(unpack_bits(out.PixelData, as_array=False))[:30]

        read = _roundtrip(out)
        assert 3 == read.NumberOfFrames
        assert expected == list(unpack_bits(read.PixelData, as_array=False))[:30]
        assert [5, 2, 5] == _frame_numbers(read)

    def test_odd_frame_length_padded(self):
        """Odd-length selected pixel data is padded to an even length."""
        ds = _enhanced_ds(rows=1, cols=3)
        out = select_frames(ds, [2])
        assert 1 == out.NumberOfFrames
        assert 4 == len(out.PixelData)
        assert b"\x02\x02\x02\x00" == out.PixelData


def _compressed_frames(nr_frames=5):
    """Return ``nr_frames`` fake compressed frames of different lengths."""
    return [bytes([i] * (11 + i)) for i in range(nr_frames)]


def _encapsulated_ds(frames, **kwargs):
    """Return an enhanced multi-frame dataset with encapsulated pixel data
    made up of ``frames``.
    """
    ds = _enhanced_ds(nr_frames=len(frames))
    ds.file_meta.TransferSyntaxUID = RLELossless
    ds.PixelData = encapsulate(frames, **kwargs)
    return ds


class TestSelectFramesEncapsulated:
    """Tests for encapsulated (compressed) pixel data."""

    def test_basic_offset_table(self):
        """Compressed frames are copied without decoding, BOT is rebuilt."""
        frames = _compressed_frames()
        ds = _encapsulated_ds(frames, fragments_per_frame=2, has_bot=True)
        original = copy.deepcopy(ds)
        # The frames as recoverable from the source (may include padding)
        source_frames = list(generate_frames(ds.PixelData, number_of_frames=5))

        out = ds.select_frames(INDICES)
        assert ds == original

        # The fragments of the selected frames are preserved
        expected = [source_frames[i] for i in INDICES]
        selected = list(generate_fragmented_frames(out.PixelData, number_of_frames=3))
        assert 3 == len(selected)
        for fragments, frame in zip(selected, expected):
            assert 2 == len(fragments)
            assert frame == b"".join(fragments)

        # The Basic Offset Table has been rebuilt for the selection
        assert 3 == len(parse_basic_offsets(out.PixelData))
        assert "ExtendedOffsetTable" not in out
        assert "ExtendedOffsetTableLengths" not in out

        read = _roundtrip(out)
        assert expected == list(generate_frames(read.PixelData, number_of_frames=3))
        assert [5, 2, 5] == _frame_numbers(read)

    def test_extended_offset_table(self):
        """The Extended Offset Table is rebuilt for the selected frames."""
        frames = _compressed_frames()
        pixel_data, offsets, lengths = encapsulate_extended(frames)
        ds = _encapsulated_ds(frames, has_bot=False)
        ds.PixelData = pixel_data
        ds.ExtendedOffsetTable = offsets
        ds.ExtendedOffsetTableLengths = lengths
        source_frames = list(
            generate_frames(
                ds.PixelData,
                number_of_frames=5,
                extended_offsets=(offsets, lengths),
            )
        )

        out = select_frames(ds, INDICES)

        expected = [source_frames[i] for i in INDICES]
        extended = (out.ExtendedOffsetTable, out.ExtendedOffsetTableLengths)
        assert expected == list(
            generate_frames(
                out.PixelData, number_of_frames=3, extended_offsets=extended
            )
        )
        # Basic Offset Table is empty when an Extended Offset Table is used
        assert [] == parse_basic_offsets(out.PixelData)

        read = _roundtrip(out)
        extended = (read.ExtendedOffsetTable, read.ExtendedOffsetTableLengths)
        assert expected == list(
            generate_frames(
                read.PixelData, number_of_frames=3, extended_offsets=extended
            )
        )

    def test_no_offset_tables_raises(self):
        """A clear failure when frame boundaries cannot be determined."""
        frames = _compressed_frames()
        ds = _encapsulated_ds(frames, fragments_per_frame=2, has_bot=False)

        with pytest.raises(ValueError, match="frame boundaries"):
            select_frames(ds, INDICES)


class TestSelectFramesReferences:
    """Tests for remapping of frame number references."""

    def test_unreferenced_frames_removed(self):
        """References to frames that are not selected are removed."""
        ds = _enhanced_ds()
        ds.ReferencedImageSequence[0].ReferencedFrameNumber = [2, 4]

        out = select_frames(ds, INDICES)
        # Old frame 2 -> new frame 2, old frame 4 -> not selected
        assert 2 == out.ReferencedImageSequence[0].ReferencedFrameNumber

    def test_empty_reference_removed(self):
        """The element is removed if no referenced frames remain."""
        ds = _enhanced_ds()
        ds.ReferencedImageSequence[0].ReferencedFrameNumber = [1, 3]

        out = select_frames(ds, INDICES)
        assert "ReferencedFrameNumber" not in out.ReferencedImageSequence[0]

    def test_nested_references_remapped(self):
        """References nested in sequences are remapped."""
        ds = _enhanced_ds()
        del ds.ReferencedImageSequence
        request = Dataset()
        request.ReferencedFrameNumbers = [4, 5]
        procedure = Dataset()
        procedure.ReferencedSOPSequence = [request]
        ds.ReferencedRequestSequence = [procedure]

        out = select_frames(ds, INDICES)
        # Old frames 4 and 5 -> new frames 1 and 3 (frame 4 selected twice)
        assert [1, 3] == (
            out.ReferencedRequestSequence[0]
            .ReferencedSOPSequence[0]
            .ReferencedFrameNumbers[:]
        )


class TestSelectFramesSOPInstanceUID:
    """Tests for the caller-provided SOP Instance UID policy."""

    def test_default_keeps_original(self):
        ds = _enhanced_ds()
        out = select_frames(ds, INDICES)
        assert ds.SOPInstanceUID == out.SOPInstanceUID

    def test_uid_string(self):
        ds = _enhanced_ds()
        out = select_frames(ds, INDICES, sop_instance_uid="1.2.3.4.5")
        assert "1.2.3.4.5" == out.SOPInstanceUID
        assert "1.2.3.4.5" == out.file_meta.MediaStorageSOPInstanceUID

    def test_uid_callable(self):
        ds = _enhanced_ds()
        calls = []

        def policy(new_ds, indices):
            calls.append((new_ds, indices))
            return generate_uid()

        out = select_frames(ds, INDICES, sop_instance_uid=policy)
        assert 1 == len(calls)
        assert [4, 1, 4] == calls[0][1]
        assert out.SOPInstanceUID != ds.SOPInstanceUID
        assert out.SOPInstanceUID == out.file_meta.MediaStorageSOPInstanceUID


class TestSelectFramesValidation:
    """Tests for input validation."""

    def test_empty_indices(self):
        ds = _enhanced_ds()
        with pytest.raises(ValueError, match="at least one frame index"):
            select_frames(ds, [])

    def test_negative_index(self):
        ds = _enhanced_ds()
        with pytest.raises(ValueError, match="non-negative"):
            select_frames(ds, [1, -1])

    def test_non_integer_index(self):
        ds = _enhanced_ds()
        with pytest.raises(TypeError, match="must contain integers"):
            select_frames(ds, [1, "2"])

    def test_index_out_of_range(self):
        ds = _enhanced_ds()
        with pytest.raises(IndexError, match="out of range"):
            select_frames(ds, [1, 5])

    def test_no_pixel_data(self):
        ds = Dataset()
        ds.NumberOfFrames = "5"
        with pytest.raises(AttributeError, match="Pixel Data"):
            select_frames(ds, INDICES)

    def test_per_frame_groups_mismatch(self):
        ds = _enhanced_ds()
        del ds.PerFrameFunctionalGroupsSequence[0]
        with pytest.raises(ValueError, match="one item per frame"):
            select_frames(ds, INDICES)
