# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Tests for pydicom.pixels.crop.crop_pixels."""

from copy import deepcopy

import numpy as np
import pytest

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixels import crop_pixels
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, UID, generate_uid


# Oblique (not axis-aligned) orthonormal direction cosines
ROW_COSINES = (0.36, 0.48, 0.8)
COL_COSINES = (0.48, 0.64, -0.6)
ORIENTATION = list(ROW_COSINES) + list(COL_COSINES)
POSITION = [10.0, -20.0, 30.0]
SPACING = [0.7, 0.8]  # row spacing, column spacing


def _world(position, orientation, spacing, row, col):
    """Return the patient coordinates of the center of pixel (row, col)."""
    row_cosines = np.array(orientation[:3], dtype=float)
    col_cosines = np.array(orientation[3:], dtype=float)
    return (
        np.array(position, dtype=float)
        + row_cosines * row * spacing[0]
        + col_cosines * col * spacing[1]
    )


def _single_frame_ds(rows=64, cols=48):
    """Return an oblique single frame signed 12-bit dataset and its array."""
    rng = np.random.default_rng(42)
    arr = rng.integers(-2048, 2047, size=(rows, cols), dtype=np.int16)

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")  # CT Image Storage
    ds.SOPInstanceUID = generate_uid()
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = SPACING
    ds.ImagePositionPatient = POSITION
    ds.ImageOrientationPatient = ORIENTATION
    ds.PixelData = arr.tobytes()
    return ds, arr


def _enhanced_ds(nr_frames=3, rows=32, cols=24):
    """Return an Enhanced multi-frame dataset, its frames and positions."""
    rng = np.random.default_rng(1234)
    frames = rng.integers(0, 4095, size=(nr_frames, rows, cols)).astype(np.uint16)

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2.1")  # Enhanced CT
    ds.SOPInstanceUID = generate_uid()
    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = str(nr_frames)
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    shared = Dataset()
    measures = Dataset()
    measures.PixelSpacing = SPACING
    shared.PixelMeasuresSequence = Sequence([measures])
    orientation = Dataset()
    orientation.ImageOrientationPatient = ORIENTATION
    shared.PlaneOrientationSequence = Sequence([orientation])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])

    positions = [[1.5 * i, -2.0 * i, 3.0 * i + 7.0] for i in range(nr_frames)]
    items = []
    for position in positions:
        item = Dataset()
        plane_position = Dataset()
        plane_position.ImagePositionPatient = position
        item.PlanePositionSequence = Sequence([plane_position])
        items.append(item)
    ds.PerFrameFunctionalGroupsSequence = Sequence(items)

    ds.PixelData = frames.tobytes()
    return ds, frames, positions


def _decode(ds, dtype, shape):
    return np.frombuffer(ds.PixelData, dtype=dtype).reshape(shape)


class TestCropSingleFrame:
    def test_oblique_corners(self):
        """Cropped corners of an oblique image land on the original pixels."""
        ds, arr = _single_frame_ds()
        top, left, height, width = 10, 5, 30, 20

        new = crop_pixels(ds, (top, left, height, width))

        assert new.Rows == height
        assert new.Columns == width

        # The new origin is the patient position of pixel (top, left)
        expected_origin = _world(POSITION, ORIENTATION, SPACING, top, left)
        origin = [float(v) for v in new.ImagePositionPatient]
        assert np.allclose(origin, expected_origin)

        # All four corners of the crop must coincide with the corresponding
        # original pixel centers in patient coordinates
        corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
        for row, col in corners:
            from_new = _world(origin, ORIENTATION, SPACING, row, col)
            from_old = _world(POSITION, ORIENTATION, SPACING, top + row, left + col)
            assert np.allclose(from_new, from_old)

    def test_pixels_and_bit_depth_preserved(self):
        """Cropped pixels equal the original ones, bit depth is untouched."""
        ds, arr = _single_frame_ds()
        top, left, height, width = 10, 5, 30, 20

        new = crop_pixels(ds, (top, left, height, width))

        cropped = _decode(new, np.int16, (height, width))
        assert np.array_equal(cropped, arr[top : top + height, left : left + width])
        # Signed values survive the round trip
        assert cropped.min() < 0
        # No silent conversion to an 8-bit preview
        assert new.BitsAllocated == 16
        assert new.BitsStored == 12
        assert new.HighBit == 11
        assert new.PixelRepresentation == 1

    def test_original_dataset_unchanged(self):
        ds, arr = _single_frame_ds()
        before = deepcopy(ds)

        crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before
        assert ds.Rows == 64
        assert [float(v) for v in ds.ImagePositionPatient] == POSITION

    def test_full_image_crop_keeps_position(self):
        ds, arr = _single_frame_ds()

        new = crop_pixels(ds, (0, 0, 64, 48))

        assert np.allclose([float(v) for v in new.ImagePositionPatient], POSITION)
        assert np.array_equal(_decode(new, np.int16, (64, 48)), arr)

    def test_spacing_and_orientation_unchanged(self):
        ds, arr = _single_frame_ds()

        new = crop_pixels(ds, (10, 5, 30, 20))

        assert [float(v) for v in new.PixelSpacing] == SPACING
        assert [float(v) for v in new.ImageOrientationPatient] == ORIENTATION


class TestCropEnhancedMultiframe:
    def test_shared_functional_groups(self):
        """Geometry from the shared groups shifts every frame's position."""
        ds, frames, positions = _enhanced_ds()
        top, left, height, width = 4, 3, 20, 10

        new = crop_pixels(ds, (top, left, height, width))

        assert new.Rows == height
        assert new.Columns == width
        assert new.NumberOfFrames == "3"

        offset = _world([0, 0, 0], ORIENTATION, SPACING, top, left)
        for index, position in enumerate(positions):
            item = new.PerFrameFunctionalGroupsSequence[index]
            actual = [
                float(v) for v in item.PlanePositionSequence[0].ImagePositionPatient
            ]
            assert np.allclose(actual, np.array(position) + offset)

        cropped = _decode(new, np.uint16, (3, height, width))
        assert np.array_equal(
            cropped, frames[:, top : top + height, left : left + width]
        )

    def test_per_frame_scope_overrides_shared(self):
        """A frame's own functional group values take precedence."""
        ds, frames, positions = _enhanced_ds()
        # Frame 1 has its own orientation and spacing
        frame_orientation = [0.6, 0.0, -0.8, 0.0, 1.0, 0.0]
        frame_spacing = [1.0, 2.0]
        item = ds.PerFrameFunctionalGroupsSequence[1]
        orientation = Dataset()
        orientation.ImageOrientationPatient = frame_orientation
        item.PlaneOrientationSequence = Sequence([orientation])
        measures = Dataset()
        measures.PixelSpacing = frame_spacing
        item.PixelMeasuresSequence = Sequence([measures])

        top, left, height, width = 4, 3, 20, 10
        new = crop_pixels(ds, (top, left, height, width))

        shared_offset = _world([0, 0, 0], ORIENTATION, SPACING, top, left)
        frame_offset = _world([0, 0, 0], frame_orientation, frame_spacing, top, left)
        for index, position in enumerate(positions):
            actual = [
                float(v)
                for v in new.PerFrameFunctionalGroupsSequence[index]
                .PlanePositionSequence[0]
                .ImagePositionPatient
            ]
            expected = np.array(position) + (
                frame_offset if index == 1 else shared_offset
            )
            assert np.allclose(actual, expected)

    def test_top_level_position_does_not_override_frames(self):
        """A stray top-level position must not replace per-frame positions."""
        ds, frames, positions = _enhanced_ds()
        ds.ImagePositionPatient = [999.0, 999.0, 999.0]
        top, left, height, width = 4, 3, 20, 10

        new = crop_pixels(ds, (top, left, height, width))

        offset = _world([0, 0, 0], ORIENTATION, SPACING, top, left)
        for index, position in enumerate(positions):
            actual = [
                float(v)
                for v in new.PerFrameFunctionalGroupsSequence[index]
                .PlanePositionSequence[0]
                .ImagePositionPatient
            ]
            assert np.allclose(actual, np.array(position) + offset)

    def test_original_dataset_unchanged(self):
        ds, frames, positions = _enhanced_ds()
        before = deepcopy(ds)

        crop_pixels(ds, (4, 3, 20, 10))

        assert ds == before


class TestCropErrors:
    @pytest.mark.parametrize(
        "crop_box",
        [
            (60, 0, 10, 10),  # bottom edge out of bounds
            (0, 40, 10, 10),  # right edge out of bounds
            (-1, 0, 10, 10),  # negative top
            (0, -1, 10, 10),  # negative left
            (0, 0, 0, 10),  # zero height
            (0, 0, 10, 0),  # zero width
        ],
    )
    def test_out_of_bounds(self, crop_box):
        ds, arr = _single_frame_ds()
        before = deepcopy(ds)

        with pytest.raises(ValueError, match="crop box"):
            crop_pixels(ds, crop_box)

        assert ds == before

    def test_missing_orientation(self):
        ds, arr = _single_frame_ds()
        del ds.ImageOrientationPatient
        before = deepcopy(ds)

        with pytest.raises(ValueError, match="Image Orientation Patient"):
            crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before

    def test_missing_spacing(self):
        ds, arr = _single_frame_ds()
        del ds.PixelSpacing
        before = deepcopy(ds)

        with pytest.raises(ValueError, match="Pixel Spacing"):
            crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before

    def test_missing_position(self):
        ds, arr = _single_frame_ds()
        del ds.ImagePositionPatient
        before = deepcopy(ds)

        with pytest.raises(ValueError, match="Image Position Patient"):
            crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before

    def test_non_orthogonal_orientation(self):
        ds, arr = _single_frame_ds()
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.5, 0.5, 0.0]
        before = deepcopy(ds)

        with pytest.raises(ValueError, match="not a unit vector|not orthogonal"):
            crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before

    def test_skewed_orientation(self):
        ds, arr = _single_frame_ds()
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.6, 0.8, 0.0]
        before = deepcopy(ds)

        with pytest.raises(ValueError, match="not orthogonal"):
            crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before

    def test_encapsulated_raises(self):
        ds, arr = _single_frame_ds()
        ds.file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.4.90")
        before = deepcopy(ds)

        with pytest.raises(NotImplementedError, match="compressed"):
            crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before

    def test_color_raises(self):
        ds, arr = _single_frame_ds()
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        before = deepcopy(ds)

        with pytest.raises(NotImplementedError, match="grayscale"):
            crop_pixels(ds, (10, 5, 30, 20))

        assert ds == before

    def test_enhanced_missing_frame_orientation(self):
        """One frame missing orientation everywhere must fail loudly."""
        ds, frames, positions = _enhanced_ds()
        del ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence
        before = deepcopy(ds)

        with pytest.raises(ValueError, match="Image Orientation Patient"):
            crop_pixels(ds, (4, 3, 20, 10))

        assert ds == before

    def test_invalid_crop_box_type(self):
        ds, arr = _single_frame_ds()

        with pytest.raises(ValueError, match="crop_box"):
            crop_pixels(ds, (10, 5, 30))
        with pytest.raises(ValueError, match="crop_box"):
            crop_pixels(ds, (10.5, 5, 30, 20))
