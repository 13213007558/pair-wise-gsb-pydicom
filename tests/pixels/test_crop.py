# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Tests for the pixels.crop module."""

import math

import pytest

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixels import crop
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    JPEG2000Lossless,
)

pytestmark = pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")


SQRT2 = math.sqrt(2.0)
# Oblique (non axis-aligned) orthonormal direction cosines
ROW_COSINES = [1.0 / SQRT2, 1.0 / SQRT2, 0.0]
COLUMN_COSINES = [-0.5, 0.5, 1.0 / SQRT2]
OBLIQUE_IOP = ROW_COSINES + COLUMN_COSINES
IPP = [10.0, 20.0, 30.0]
PIXEL_SPACING = [0.5, 0.25]  # (row spacing, column spacing)


def _meta(tsyntax=ExplicitVRLittleEndian):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = tsyntax
    return meta


def _world_position(ipp, iop, spacing, row, column):
    """Return the patient coordinates of the center of pixel (row, column)."""
    return [
        ipp[i] + iop[i] * column * spacing[1] + iop[i + 3] * row * spacing[0]
        for i in range(3)
    ]


def _single_frame(tsyntax=ExplicitVRLittleEndian, rows=6, columns=8):
    """Return a 16-bit signed single frame dataset with oblique geometry."""
    ds = Dataset()
    ds.file_meta = _meta(tsyntax)
    ds.Rows = rows
    ds.Columns = columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1
    ds.ImagePositionPatient = list(IPP)
    ds.ImageOrientationPatient = list(OBLIQUE_IOP)
    ds.PixelSpacing = list(PIXEL_SPACING)
    endian = ">" if tsyntax == ExplicitVRBigEndian else "<"
    arr = (np.arange(rows * columns, dtype=np.int32) - 20).astype(f"{endian}i2")
    ds.add_new(0x7FE00010, "OW", arr.tobytes())
    return ds, arr


def _enhanced_multiframe():
    """Return a 3-frame 16-bit unsigned enhanced multi-frame dataset."""
    ds = Dataset()
    ds.file_meta = _meta()
    ds.Rows = 4
    ds.Columns = 5
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = "3"
    # Bogus top-level position that must not override the per-frame ones
    ds.ImagePositionPatient = [999.0, 999.0, 999.0]

    shared = Dataset()
    shared.PixelMeasuresSequence = Sequence([Dataset()])
    shared.PixelMeasuresSequence[0].PixelSpacing = [2.0, 3.0]
    shared.PlaneOrientationSequence = Sequence([Dataset()])
    shared.PlaneOrientationSequence[0].ImageOrientationPatient = list(OBLIQUE_IOP)
    ds.SharedFunctionalGroupsSequence = Sequence([shared])

    positions = [[0.0, 0.0, 100.0], [1.0, 0.0, 100.0], [2.0, 0.0, 100.0]]
    per_frame = []
    for position in positions:
        item = Dataset()
        item.PlanePositionSequence = Sequence([Dataset()])
        item.PlanePositionSequence[0].ImagePositionPatient = position
        per_frame.append(item)
    # Frame 1 uses its own measures and orientation instead of the shared ones
    per_frame[1].PixelMeasuresSequence = Sequence([Dataset()])
    per_frame[1].PixelMeasuresSequence[0].PixelSpacing = [1.0, 1.5]
    per_frame[1].PlaneOrientationSequence = Sequence([Dataset()])
    per_frame[1].PlaneOrientationSequence[0].ImageOrientationPatient = [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)

    arr = np.arange(3 * 4 * 5, dtype="<u2")
    ds.add_new(0x7FE00010, "OW", arr.tobytes())
    return ds, arr, positions


class TestCropSingleFrame:
    """Tests for cropping single frame datasets."""

    def test_oblique_corners_world_position(self):
        """The corners of the crop keep their original world coordinates."""
        ds, arr = _single_frame()
        out = crop(ds, (2, 1, 6, 5))

        assert out.Rows == 4
        assert out.Columns == 4

        new_ipp = [float(v) for v in out.ImagePositionPatient]
        expected_ipp = _world_position(IPP, OBLIQUE_IOP, PIXEL_SPACING, 1, 2)
        assert new_ipp == pytest.approx(expected_ipp, abs=1e-6)

        # All four corners of the cropped image are located at the patient
        #   coordinates of the corresponding pixels in the original image
        for r, c in ((0, 0), (0, 3), (3, 0), (3, 3)):
            corner = _world_position(new_ipp, OBLIQUE_IOP, PIXEL_SPACING, r, c)
            reference = _world_position(IPP, OBLIQUE_IOP, PIXEL_SPACING, r + 1, c + 2)
            assert corner == pytest.approx(reference, abs=1e-6)

    def test_axis_aligned(self):
        """Axis-aligned geometry is translated along the patient axes."""
        ds, arr = _single_frame()
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        out = crop(ds, (3, 2, 8, 6))

        ipp = [float(v) for v in out.ImagePositionPatient]
        assert ipp == pytest.approx([10.0 + 3 * 0.25, 20.0 + 2 * 0.5, 30.0])

    def test_pixel_values_and_bit_depth_preserved(self):
        """Pixels are copied 1:1 with no conversion of the bit depth."""
        ds, arr = _single_frame()
        out = crop(ds, (2, 1, 6, 5))

        result = np.frombuffer(out.PixelData, dtype="<i2").reshape(4, 4)
        expected = arr.reshape(6, 8)[1:5, 2:6]
        assert np.array_equal(result, expected)
        # Signed 16-bit data with negative values, not an 8-bit preview
        assert result.min() < 0
        assert out.BitsAllocated == 16
        assert out.BitsStored == 12
        assert out.HighBit == 11
        assert out.PixelRepresentation == 1
        assert out[0x7FE00010].VR == "OW"

    def test_big_endian(self):
        """Big endian pixel data stays big endian."""
        ds, arr = _single_frame(tsyntax=ExplicitVRBigEndian)
        out = crop(ds, (2, 1, 6, 5))

        result = np.frombuffer(out.PixelData, dtype=">i2").reshape(4, 4)
        assert np.array_equal(result, arr.reshape(6, 8)[1:5, 2:6])

    def test_full_image_box(self):
        """Cropping to the full image leaves position and pixels unchanged."""
        ds, arr = _single_frame()
        out = crop(ds, (0, 0, 8, 6))

        assert [float(v) for v in out.ImagePositionPatient] == pytest.approx(IPP)
        assert np.array_equal(np.frombuffer(out.PixelData, dtype="<i2"), arr)

    def test_single_pixel_box(self):
        """A 1x1 crop works and lands on the corresponding pixel."""
        ds, arr = _single_frame()
        out = crop(ds, (7, 5, 8, 6))

        assert out.Rows == 1
        assert out.Columns == 1
        result = np.frombuffer(out.PixelData, dtype="<i2")
        assert result[0] == arr.reshape(6, 8)[5, 7]
        ipp = [float(v) for v in out.ImagePositionPatient]
        assert ipp == pytest.approx(
            _world_position(IPP, OBLIQUE_IOP, PIXEL_SPACING, 5, 7)
        )

    def test_original_dataset_unchanged(self):
        """Cropping returns a new dataset and doesn't modify the original."""
        ds, arr = _single_frame()
        snapshot = str(ds)
        out = crop(ds, (2, 1, 6, 5))

        assert out is not ds
        assert str(ds) == snapshot
        assert ds.Rows == 6
        assert ds.Columns == 8
        assert len(ds.PixelData) == 6 * 8 * 2


class TestCropEnhancedMultiframe:
    """Tests for cropping enhanced multi-frame datasets."""

    def test_per_frame_positions(self):
        """Each frame position is translated using its own scope's values."""
        ds, arr, positions = _enhanced_multiframe()
        out = crop(ds, (1, 1, 4, 3))

        assert out.Rows == 2
        assert out.Columns == 3
        assert out.NumberOfFrames == "3"

        frames = out.PerFrameFunctionalGroupsSequence
        # Frames 0 and 2 use the shared measures and orientation
        for index in (0, 2):
            ipp = [
                float(v)
                for v in frames[index].PlanePositionSequence[0].ImagePositionPatient
            ]
            expected = _world_position(positions[index], OBLIQUE_IOP, [2.0, 3.0], 1, 1)
            assert ipp == pytest.approx(expected)

        # Frame 1 uses its own per-frame measures and orientation
        ipp = [
            float(v) for v in frames[1].PlanePositionSequence[0].ImagePositionPatient
        ]
        expected = _world_position(
            positions[1], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 1.5], 1, 1
        )
        assert ipp == pytest.approx(expected)

        # The bogus top-level position is not used and left unchanged
        assert [float(v) for v in out.ImagePositionPatient] == [999.0] * 3

    def test_frame_pixels(self):
        """Every frame's pixels are cropped 1:1."""
        ds, arr, positions = _enhanced_multiframe()
        out = crop(ds, (1, 1, 4, 3))

        result = np.frombuffer(out.PixelData, dtype="<u2").reshape(3, 2, 3)
        expected = arr.reshape(3, 4, 5)[:, 1:3, 1:4]
        assert np.array_equal(result, expected)
        assert out.BitsAllocated == 16
        assert out.PixelRepresentation == 0

    def test_original_dataset_unchanged(self):
        ds, arr, positions = _enhanced_multiframe()
        snapshot = str(ds)
        crop(ds, (1, 1, 4, 3))
        assert str(ds) == snapshot

    def test_missing_per_frame_position_raises(self):
        """A frame without a Plane Position Sequence is a clear error."""
        ds, arr, positions = _enhanced_multiframe()
        del ds.PerFrameFunctionalGroupsSequence[1]["PlanePositionSequence"]
        snapshot = str(ds)
        msg = "per-frame functional groups item 1"
        with pytest.raises(ValueError, match=msg):
            crop(ds, (1, 1, 4, 3))
        assert str(ds) == snapshot

    def test_frame_count_mismatch_raises(self):
        ds, arr, positions = _enhanced_multiframe()
        ds.NumberOfFrames = "2"
        ds.PixelData = arr[: 2 * 4 * 5].tobytes()
        snapshot = str(ds)
        with pytest.raises(ValueError, match="NumberOfFrames"):
            crop(ds, (1, 1, 4, 3))
        assert str(ds) == snapshot

    def test_missing_shared_orientation_raises(self):
        ds, arr, positions = _enhanced_multiframe()
        del ds.SharedFunctionalGroupsSequence[0]["PlaneOrientationSequence"]
        del ds.PerFrameFunctionalGroupsSequence[1]["PlaneOrientationSequence"]
        snapshot = str(ds)
        with pytest.raises(ValueError, match="ImageOrientationPatient"):
            crop(ds, (1, 1, 4, 3))
        assert str(ds) == snapshot

    def test_missing_shared_spacing_raises(self):
        ds, arr, positions = _enhanced_multiframe()
        del ds.SharedFunctionalGroupsSequence[0]["PixelMeasuresSequence"]
        del ds.PerFrameFunctionalGroupsSequence[1]["PixelMeasuresSequence"]
        snapshot = str(ds)
        with pytest.raises(ValueError, match="PixelSpacing"):
            crop(ds, (1, 1, 4, 3))
        assert str(ds) == snapshot


class TestCropErrors:
    """Tests for clear errors that leave the original dataset unchanged."""

    @pytest.mark.parametrize(
        "box",
        [
            (-1, 0, 4, 4),
            (0, -1, 4, 4),
            (0, 0, 9, 6),
            (0, 0, 8, 7),
            (2, 1, 2, 5),
            (2, 1, 6, 1),
            (9, 0, 10, 2),
        ],
    )
    def test_out_of_bounds_box_raises(self, box):
        ds, arr = _single_frame()
        snapshot = str(ds)
        with pytest.raises(ValueError, match="out of bounds"):
            crop(ds, box)
        assert str(ds) == snapshot

    def test_invalid_box_raises(self):
        ds, arr = _single_frame()
        with pytest.raises(ValueError, match="4 integers"):
            crop(ds, (1, 2, 3))

    @pytest.mark.parametrize(
        "keyword",
        [
            "Rows",
            "Columns",
            "BitsAllocated",
            "PixelSpacing",
            "ImagePositionPatient",
            "ImageOrientationPatient",
        ],
    )
    def test_missing_required_element_raises(self, keyword):
        ds, arr = _single_frame()
        del ds[keyword]
        snapshot = str(ds)
        with pytest.raises(ValueError, match=keyword):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_missing_pixel_data_raises(self):
        ds, arr = _single_frame()
        del ds["PixelData"]
        snapshot = str(ds)
        with pytest.raises(ValueError, match="Pixel Data"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_compressed_transfer_syntax_raises(self):
        ds, arr = _single_frame()
        ds.file_meta.TransferSyntaxUID = JPEG2000Lossless
        snapshot = str(ds)
        with pytest.raises(ValueError, match="compressed transfer syntaxes"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_color_samples_raise(self):
        ds, arr = _single_frame()
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        snapshot = str(ds)
        with pytest.raises(ValueError, match="grayscale"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_non_monochrome_raises(self):
        ds, arr = _single_frame()
        ds.PhotometricInterpretation = "PALETTE COLOR"
        snapshot = str(ds)
        with pytest.raises(ValueError, match="MONOCHROME"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_bit_packed_raises(self):
        ds, arr = _single_frame()
        ds.BitsAllocated = 1
        ds.BitsStored = 1
        ds.HighBit = 0
        snapshot = str(ds)
        with pytest.raises(ValueError, match="bit-packed"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_non_unit_cosines_raise(self):
        ds, arr = _single_frame()
        ds.ImageOrientationPatient = [2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        snapshot = str(ds)
        with pytest.raises(ValueError, match="unit vector"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_non_orthogonal_cosines_raise(self):
        ds, arr = _single_frame()
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.5, math.sqrt(0.75), 0.0]
        snapshot = str(ds)
        with pytest.raises(ValueError, match="orthogonal"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_zero_spacing_raises(self):
        ds, arr = _single_frame()
        ds.PixelSpacing = [0.5, 0.0]
        snapshot = str(ds)
        with pytest.raises(ValueError, match="greater than zero"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_pixel_data_length_mismatch_raises(self):
        ds, arr = _single_frame()
        ds.PixelData = ds.PixelData + bytes(2)
        snapshot = str(ds)
        with pytest.raises(ValueError, match="doesn't match"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot

    def test_unknown_encoding_raises(self):
        ds, arr = _single_frame()
        del ds.file_meta
        snapshot = str(ds)
        with pytest.raises(ValueError, match="Transfer Syntax"):
            crop(ds, (0, 0, 4, 4))
        assert str(ds) == snapshot
