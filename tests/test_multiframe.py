"""Tests for pydicom.multiframe.select_frames and Dataset.select_frames."""

from io import BytesIO

import pytest

from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import (
    encapsulate,
    encapsulate_extended,
    generate_frames,
    parse_basic_offsets,
)
from pydicom.multiframe import select_frames
from pydicom.pixels import pack_bits, unpack_bits
from pydicom.uid import (
    EnhancedMRImageStorage,
    ExplicitVRLittleEndian,
    RLELossless,
    generate_uid,
)


NEW_UID = "1.2.826.0.1.3680043.9.7433.1.1"


def _per_frame_item(index, sop_instance_uid):
    """Return a per-frame functional groups item for zero-based `index`."""
    item = Dataset()
    plane_position = Dataset()
    plane_position.ImagePositionPatient = [float(index), 0.0, 0.0]
    item.PlanePositionSequence = [plane_position]

    frame_content = Dataset()
    frame_content.DimensionIndexValues = [1, index + 1]
    item.FrameContentSequence = [frame_content]

    # Each item references its own frame number (1-based)
    source_image = Dataset()
    source_image.ReferencedSOPClassUID = EnhancedMRImageStorage
    source_image.ReferencedSOPInstanceUID = sop_instance_uid
    source_image.ReferencedFrameNumber = index + 1
    derivation = Dataset()
    derivation.SourceImageSequence = [source_image]
    item.DerivationImageSequence = [derivation]

    return item


def _make_enhanced_ds(nr_frames=6, rows=4, columns=5, bits_allocated=16):
    """Return an enhanced multi-frame dataset with native pixel data."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
    ds.SOPClassUID = EnhancedMRImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.PatientName = "Frame^Select"
    ds.PatientID = "123456"
    ds.Modality = "MR"
    ds.Rows = rows
    ds.Columns = columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated
    ds.HighBit = bits_allocated - 1
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = nr_frames

    # Shared functional groups keep their shared semantics
    pixel_measures = Dataset()
    pixel_measures.SliceThickness = 2.5
    pixel_measures.PixelSpacing = [0.5, 0.5]
    shared = Dataset()
    shared.PixelMeasuresSequence = [pixel_measures]
    ds.SharedFunctionalGroupsSequence = [shared]

    ds.PerFrameFunctionalGroupsSequence = [
        _per_frame_item(ii, ds.SOPInstanceUID) for ii in range(nr_frames)
    ]

    # Top-level references to frames of the same object
    ref_kept = Dataset()
    ref_kept.ReferencedSOPClassUID = EnhancedMRImageStorage
    ref_kept.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    ref_kept.ReferencedFrameNumber = [2, 5]
    ref_dropped = Dataset()
    ref_dropped.ReferencedSOPClassUID = EnhancedMRImageStorage
    ref_dropped.ReferencedSOPInstanceUID = ds.SOPInstanceUID
    ref_dropped.ReferencedFrameNumber = 3
    ds.ReferencedImageSequence = [ref_kept, ref_dropped]

    if bits_allocated > 1:
        bytes_per_pixel = bits_allocated // 8
        ds.PixelData = b"".join(
            (ii + 1).to_bytes(bytes_per_pixel, "little") * (rows * columns)
            for ii in range(nr_frames)
        )

    return ds


def _roundtrip(ds):
    """Write `ds` to a buffer and read it back."""
    buffer = BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    buffer.seek(0)
    return dcmread(buffer)


def _frame_values(ds, index):
    """Return the pixel values of frame `index` of a native dataset."""
    bytes_per_frame = ds.Rows * ds.Columns * (ds.BitsAllocated // 8)
    start = index * bytes_per_frame
    return ds.PixelData[start : start + bytes_per_frame]


class TestSelectFramesNative:
    """Tests for frame selection with native (uncompressed) pixel data."""

    def test_order_and_duplicates_preserved(self):
        """Frames, functional groups and references all follow [4, 1, 4]."""
        ds = _make_enhanced_ds()
        original_pixel_data = ds.PixelData

        out = ds.select_frames([4, 1, 4], sop_instance_uid=NEW_UID)

        # The input dataset is unchanged
        assert int(ds.NumberOfFrames) == 6
        assert len(ds.PerFrameFunctionalGroupsSequence) == 6
        assert ds.PixelData == original_pixel_data
        assert ds.SOPInstanceUID != NEW_UID

        assert int(out.NumberOfFrames) == 3
        assert out.SOPInstanceUID == NEW_UID
        assert out.file_meta.MediaStorageSOPInstanceUID == NEW_UID

        reread = _roundtrip(out)
        assert int(reread.NumberOfFrames) == 3

        # Pixel data corresponds item-by-item to [4, 1, 4]
        assert _frame_values(reread, 0) == _frame_values(ds, 4)
        assert _frame_values(reread, 1) == _frame_values(ds, 1)
        assert _frame_values(reread, 2) == _frame_values(ds, 4)

        # Per-frame functional groups correspond item-by-item to [4, 1, 4]
        per_frame = reread.PerFrameFunctionalGroupsSequence
        assert len(per_frame) == 3
        positions = [
            item.PlanePositionSequence[0].ImagePositionPatient[0] for item in per_frame
        ]
        assert [float(v) for v in positions] == [4.0, 1.0, 4.0]
        indices = [
            list(item.FrameContentSequence[0].DimensionIndexValues)
            for item in per_frame
        ]
        assert indices == [[1, 5], [1, 2], [1, 5]]

        # Frame references are remapped to the new frame positions
        references = [
            int(
                item.DerivationImageSequence[0]
                .SourceImageSequence[0]
                .ReferencedFrameNumber
            )
            for item in per_frame
        ]
        assert references == [1, 2, 1]

        # Top-level references are remapped as well; references to frames
        #   that weren't selected are removed
        kept, dropped = reread.ReferencedImageSequence
        assert [int(v) for v in kept.ReferencedFrameNumber] == [2, 1]
        assert "ReferencedFrameNumber" not in dropped

        # Shared functional groups are unchanged
        shared = reread.SharedFunctionalGroupsSequence[0]
        assert float(shared.PixelMeasuresSequence[0].SliceThickness) == 2.5

    def test_module_level_function(self):
        """The module-level select_frames() matches the Dataset method."""
        ds = _make_enhanced_ds()
        out = select_frames(ds, [4, 1, 4], sop_instance_uid=NEW_UID)
        assert int(out.NumberOfFrames) == 3
        assert _frame_values(out, 0) == _frame_values(ds, 4)

    def test_sop_instance_uid_policies(self):
        """The output SOP Instance UID follows the caller's policy."""
        ds = _make_enhanced_ds()

        # Plain string policy
        out = ds.select_frames([0], sop_instance_uid=NEW_UID)
        assert out.SOPInstanceUID == NEW_UID

        # Callable policy receives the new dataset
        seen = []

        def policy(new_ds):
            seen.append(int(new_ds.NumberOfFrames))
            return NEW_UID

        out = ds.select_frames([2, 3], sop_instance_uid=policy)
        assert out.SOPInstanceUID == NEW_UID
        assert seen == [2]

        # Default policy generates a fresh UID
        out = ds.select_frames([0])
        assert out.SOPInstanceUID != ds.SOPInstanceUID
        assert out.file_meta.MediaStorageSOPInstanceUID == out.SOPInstanceUID

    def test_negative_indices(self):
        """Negative indices count back from the last frame."""
        ds = _make_enhanced_ds()
        out = ds.select_frames([-1, 0])
        assert _frame_values(out, 0) == _frame_values(ds, 5)
        assert _frame_values(out, 1) == _frame_values(ds, 0)

    def test_bits_allocated_8(self):
        """8-bit pixel data is sliced per frame."""
        ds = _make_enhanced_ds(bits_allocated=8)
        out = ds.select_frames([4, 1, 4])
        assert _frame_values(out, 0) == _frame_values(ds, 4)
        assert _frame_values(out, 1) == _frame_values(ds, 1)
        assert _frame_values(out, 2) == _frame_values(ds, 4)

    def test_bit_packed_single_bit(self):
        """Bit-packed (Bits Allocated = 1) frames are extracted correctly."""
        rows, columns = 2, 3  # 6 pixels per frame, not byte aligned
        nr_pixels = rows * columns
        ds = _make_enhanced_ds(rows=rows, columns=columns, bits_allocated=1)

        def frame_bits(index):
            return [(index + jj) % 2 for jj in range(nr_pixels)]

        all_bits = [bit for ii in range(6) for bit in frame_bits(ii)]
        ds.PixelData = pack_bits(bytes(all_bits))

        out = ds.select_frames([4, 1, 4], sop_instance_uid=NEW_UID)
        assert int(out.NumberOfFrames) == 3

        reread = _roundtrip(out)
        unpacked = list(unpack_bits(reread.PixelData, as_array=False))
        expected = frame_bits(4) + frame_bits(1) + frame_bits(4)
        assert unpacked[: len(expected)] == expected

        # The input dataset is unchanged
        assert list(unpack_bits(ds.PixelData, as_array=False))[: len(all_bits)] == (
            all_bits
        )

    def test_empty_indices_raises(self):
        ds = _make_enhanced_ds()
        with pytest.raises(ValueError, match="at least one frame index"):
            ds.select_frames([])

    def test_index_out_of_range_raises(self):
        ds = _make_enhanced_ds()
        with pytest.raises(IndexError, match="out of range"):
            ds.select_frames([6])
        with pytest.raises(IndexError, match="out of range"):
            ds.select_frames([-7])

    def test_inconsistent_per_frame_sequence_raises(self):
        ds = _make_enhanced_ds()
        del ds.PerFrameFunctionalGroupsSequence[0]
        with pytest.raises(ValueError, match="inconsistent dataset"):
            ds.select_frames([0])

    def test_no_pixel_data_raises(self):
        ds = _make_enhanced_ds()
        del ds.PixelData
        with pytest.raises(ValueError, match="no 'Pixel Data'"):
            ds.select_frames([0])


def _make_encapsulated_ds(nr_frames=6, use_extended=False, empty_bot=False):
    """Return an enhanced multi-frame dataset with encapsulated pixel data."""
    frames = [bytes([ii + 1] * (10 + 2 * ii)) for ii in range(nr_frames)]

    ds = _make_enhanced_ds(nr_frames=nr_frames)
    ds.file_meta.TransferSyntaxUID = RLELossless

    if use_extended:
        ds.PixelData, ds.ExtendedOffsetTable, ds.ExtendedOffsetTableLengths = (
            encapsulate_extended(frames)
        )
    else:
        ds.PixelData = encapsulate(frames, fragments_per_frame=2, has_bot=not empty_bot)

    elem = ds["PixelData"]
    elem.is_undefined_length = True
    elem.VR = "OB"

    return ds, frames


class TestSelectFramesEncapsulated:
    """Tests for frame selection with encapsulated (compressed) pixel data."""

    def test_basic_offset_table_rebuilt(self):
        """Compressed frames are copied and the BOT rebuilt, no decoding."""
        ds, frames = _make_encapsulated_ds()
        original_pixel_data = ds.PixelData

        out = ds.select_frames([4, 1, 4], sop_instance_uid=NEW_UID)

        # The input dataset is unchanged
        assert ds.PixelData == original_pixel_data
        assert int(ds.NumberOfFrames) == 6

        assert int(out.NumberOfFrames) == 3
        assert "ExtendedOffsetTable" not in out
        assert "ExtendedOffsetTableLengths" not in out

        # The selected compressed frames are copied as-is
        selected = list(generate_frames(out.PixelData, number_of_frames=3))
        assert selected == [frames[4], frames[1], frames[4]]

        # The Basic Offset Table is rebuilt for the new frames
        offsets = parse_basic_offsets(BytesIO(out.PixelData))
        assert len(offsets) == 3
        assert offsets[0] == 0

        reread = _roundtrip(out)
        assert int(reread.NumberOfFrames) == 3
        selected = list(generate_frames(reread.PixelData, number_of_frames=3))
        assert selected == [frames[4], frames[1], frames[4]]

        # Functional groups and references still correspond to [4, 1, 4]
        per_frame = reread.PerFrameFunctionalGroupsSequence
        assert len(per_frame) == 3
        references = [
            int(
                item.DerivationImageSequence[0]
                .SourceImageSequence[0]
                .ReferencedFrameNumber
            )
            for item in per_frame
        ]
        assert references == [1, 2, 1]

    def test_extended_offset_table_rebuilt(self):
        """The Extended Offset Table is rebuilt when present."""
        ds, frames = _make_encapsulated_ds(use_extended=True)

        out = ds.select_frames([4, 1, 4], sop_instance_uid=NEW_UID)

        assert int(out.NumberOfFrames) == 3
        assert "ExtendedOffsetTable" in out
        assert "ExtendedOffsetTableLengths" in out
        assert len(out.ExtendedOffsetTable) == 3 * 8
        assert len(out.ExtendedOffsetTableLengths) == 3 * 8

        extended_offsets = (out.ExtendedOffsetTable, out.ExtendedOffsetTableLengths)
        selected = list(
            generate_frames(
                out.PixelData,
                number_of_frames=3,
                extended_offsets=extended_offsets,
            )
        )
        assert selected == [frames[4], frames[1], frames[4]]

        reread = _roundtrip(out)
        extended_offsets = (
            reread.ExtendedOffsetTable,
            reread.ExtendedOffsetTableLengths,
        )
        selected = list(
            generate_frames(
                reread.PixelData,
                number_of_frames=3,
                extended_offsets=extended_offsets,
            )
        )
        assert selected == [frames[4], frames[1], frames[4]]

    def test_undeterminable_frame_boundaries_raises(self):
        """An empty BOT with no Extended Offset Table is clearly refused."""
        ds, _ = _make_encapsulated_ds(empty_bot=True)
        assert parse_basic_offsets(BytesIO(ds.PixelData)) == []

        with pytest.raises(ValueError, match="frame boundaries"):
            ds.select_frames([4, 1, 4])
