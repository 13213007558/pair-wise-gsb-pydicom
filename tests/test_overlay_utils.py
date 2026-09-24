"""Tests for the overlays.utils module."""

import pytest

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    JPEGLosslessSV1,
)

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

try:
    from pydicom.overlays.utils import extract_overlay
except ImportError:
    pass


pytestmark = pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")


def make_ds(
    rows,
    columns,
    frames=1,
    bits_allocated=16,
    bits_stored=12,
    pixel_representation=0,
    tsyntax=ExplicitVRLittleEndian,
):
    """Return a dataset with image pixel module elements (no pixel data)."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = tsyntax
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = pixel_representation
    if frames > 1:
        ds.NumberOfFrames = str(frames)

    return ds


def add_overlay(
    ds,
    group,
    rows,
    columns,
    bit_position,
    bits_allocated=16,
    origin=None,
    frame_origin=None,
    nr_frames=None,
):
    """Add overlay plane module elements for an overlay in the pixel data."""
    ds.add_new((group, 0x0010), "US", rows)  # Overlay Rows
    ds.add_new((group, 0x0011), "US", columns)  # Overlay Columns
    ds.add_new((group, 0x0100), "US", bits_allocated)  # Overlay Bits Allocated
    ds.add_new((group, 0x0102), "US", bit_position)  # Overlay Bit Position
    if origin is not None:
        ds.add_new((group, 0x0050), "SS", origin)  # Overlay Origin
    if frame_origin is not None:
        ds.add_new((group, 0x0051), "US", frame_origin)  # Overlay Frame Origin
    if nr_frames is not None:
        # Number of Frames in Overlay
        ds.add_new((group, 0x0015), "IS", nr_frames)


def valid_values(arr, bits_stored):
    """Return the valid (stored) bits of a pixel array."""
    return np.bitwise_and(arr.astype(np.int64), (1 << bits_stored) - 1)


class TestExtractOverlay:
    """Tests for extract_overlay()."""

    def test_odd_width(self):
        """Test with an image width that isn't a multiple of 8."""
        rng = np.random.default_rng(1234)
        values = rng.integers(0, 2**12, size=(11, 13), dtype=np.uint16)
        overlay = rng.integers(0, 2, size=(11, 13), dtype=np.uint16)
        pixels = values | (overlay << 15)

        ds = make_ds(11, 13)
        ds.PixelData = pixels.tobytes()
        add_overlay(ds, 0x6000, 11, 13, 15)

        before = ds.pixel_array.copy()
        mask = extract_overlay(ds, 0x6000)

        assert mask.shape == (11, 13)
        assert np.array_equal(mask, overlay)
        # The overlay plane elements have been updated
        assert ds[(0x6000, 0x0100)].value == 1
        assert ds[(0x6000, 0x0102)].value == 0
        # The standalone overlay data round-trips
        assert np.array_equal(ds.overlay_array(0x6000), overlay)
        # The valid pixel values are unchanged
        assert np.array_equal(
            valid_values(ds.pixel_array, 12), valid_values(before, 12)
        )
        # Only the overlay bit was cleared from the pixel data
        after = np.frombuffer(ds.PixelData, dtype="<u2").reshape(11, 13)
        assert np.array_equal(after, values)

    def test_offset_origin(self):
        """Test an overlay smaller than the image with a non-(1,1) origin."""
        rng = np.random.default_rng(1234)
        values = rng.integers(0, 2**12, size=(11, 13), dtype=np.uint16)
        region = rng.integers(0, 2, size=(4, 5), dtype=np.uint16)
        pixels = values.copy()
        pixels[2:6, 3:8] |= region << 15

        ds = make_ds(11, 13)
        ds.PixelData = pixels.tobytes()
        add_overlay(ds, 0x6000, 4, 5, 15, origin=[3, 4])

        mask = extract_overlay(ds, 0x6000)

        assert mask.shape == (4, 5)
        assert np.array_equal(mask, region)
        assert np.array_equal(ds.overlay_array(0x6000), region)
        # Bits outside the overlay region are left alone
        after = np.frombuffer(ds.PixelData, dtype="<u2").reshape(11, 13)
        assert np.array_equal(after, values)

    def test_negative_origin(self):
        """Test an overlay plane extending above and left of the image."""
        rng = np.random.default_rng(1234)
        values = rng.integers(0, 2**12, size=(8, 8), dtype=np.uint16)
        region = rng.integers(0, 2, size=(3, 2), dtype=np.uint16)
        pixels = values.copy()
        pixels[0:3, 0:2] |= region << 15

        ds = make_ds(8, 8)
        ds.PixelData = pixels.tobytes()
        add_overlay(ds, 0x6000, 6, 6, 15, origin=[-2, -3])

        mask = extract_overlay(ds, 0x6000)

        assert mask.shape == (6, 6)
        expected = np.zeros((6, 6), dtype=np.uint8)
        expected[3:6, 4:6] = region
        assert np.array_equal(mask, expected)
        assert np.array_equal(ds.overlay_array(0x6000), expected)

    def test_multiframe(self):
        """Test a multi-frame overlay not starting at the first frame."""
        rng = np.random.default_rng(1234)
        values = rng.integers(0, 2**12, size=(3, 8, 8), dtype=np.uint16)
        overlays = rng.integers(0, 2, size=(3, 8, 8), dtype=np.uint16)
        pixels = values | (overlays << 15)

        ds = make_ds(8, 8, frames=3)
        ds.PixelData = pixels.tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15, frame_origin=2, nr_frames=2)

        mask = extract_overlay(ds, 0x6000)

        assert mask.shape == (2, 8, 8)
        assert np.array_equal(mask, overlays[1:3])
        assert np.array_equal(ds.overlay_array(0x6000), overlays[1:3])

        after = np.frombuffer(ds.PixelData, dtype="<u2").reshape(3, 8, 8)
        # The first frame isn't covered by the overlay and is unchanged
        assert np.array_equal(after[0], pixels[0])
        assert np.array_equal(after[1:], values[1:])

    def test_signed_pixels(self):
        """Test the signed pixel value encoding is preserved."""
        rng = np.random.default_rng(1234)
        values = rng.integers(-2000, 2000, size=(11, 13), dtype=np.int16)
        overlay = rng.integers(0, 2, size=(11, 13), dtype=np.uint16)
        pixels = (values.astype(np.uint16) & 0x0FFF) | (overlay << 15)

        ds = make_ds(11, 13, pixel_representation=1)
        ds.PixelData = pixels.tobytes()
        add_overlay(ds, 0x6000, 11, 13, 15)

        before = ds.pixel_array.copy()
        mask = extract_overlay(ds, 0x6000)

        assert np.array_equal(mask, overlay)
        # The valid pixel values are unchanged
        assert np.array_equal(
            valid_values(ds.pixel_array, 12), valid_values(before, 12)
        )
        # The 12-bit two's complement values are preserved
        expected = (values.astype(np.int32) << 20) >> 20
        actual = (ds.pixel_array.astype(np.int32) << 20) >> 20
        assert np.array_equal(actual, expected)

    def test_other_overlay_bits_unchanged(self):
        """Test bits belonging to other overlays are not cleared."""
        rng = np.random.default_rng(1234)
        values = rng.integers(0, 2**12, size=(8, 8), dtype=np.uint16)
        overlay_a = rng.integers(0, 2, size=(8, 8), dtype=np.uint16)  # bit 15
        overlay_b = rng.integers(0, 2, size=(8, 8), dtype=np.uint16)  # bit 14
        pixels = values | (overlay_a << 15) | (overlay_b << 14)

        ds = make_ds(8, 8)
        ds.PixelData = pixels.tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15)
        add_overlay(ds, 0x6002, 8, 8, 14)

        mask_a = extract_overlay(ds, 0x6000)
        assert np.array_equal(mask_a, overlay_a)

        # The second overlay's bits are still in the pixel data
        after = np.frombuffer(ds.PixelData, dtype="<u2").reshape(8, 8)
        assert np.array_equal(after, values | (overlay_b << 14))

        # And can still be extracted afterwards
        mask_b = extract_overlay(ds, 0x6002)
        assert np.array_equal(mask_b, overlay_b)
        after = np.frombuffer(ds.PixelData, dtype="<u2").reshape(8, 8)
        assert np.array_equal(after, values)

    def test_big_endian(self):
        """Test with big endian pixel data."""
        rng = np.random.default_rng(1234)
        values = rng.integers(0, 2**12, size=(8, 8), dtype=np.uint16)
        overlay = rng.integers(0, 2, size=(8, 8), dtype=np.uint16)
        pixels = values | (overlay << 15)

        ds = make_ds(8, 8, tsyntax=ExplicitVRBigEndian)
        ds.PixelData = pixels.astype(">u2").tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15)

        mask = extract_overlay(ds, 0x6000)

        assert np.array_equal(mask, overlay)
        after = np.frombuffer(ds.PixelData, dtype=">u2").reshape(8, 8)
        assert np.array_equal(after, values)

    def test_8bit_odd_length(self):
        """Test 8-bit pixel data with an odd length keeps its padding."""
        rng = np.random.default_rng(1234)
        values = rng.integers(0, 2**6, size=(11, 13), dtype=np.uint8)
        overlay = rng.integers(0, 2, size=(11, 13), dtype=np.uint8)
        pixels = values | (overlay << 7)

        ds = make_ds(11, 13, bits_allocated=8, bits_stored=6)
        ds.PixelData = pixels.tobytes() + b"\x00"  # even length padding
        add_overlay(ds, 0x6000, 11, 13, 7, bits_allocated=8)

        mask = extract_overlay(ds, 0x6000)

        assert np.array_equal(mask, overlay)
        assert len(ds.PixelData) == 11 * 13 + 1
        assert ds.PixelData[-1:] == b"\x00"
        after = np.frombuffer(ds.PixelData[: 11 * 13], dtype=np.uint8)
        assert np.array_equal(after.reshape(11, 13), values)

    def test_bit_position_in_pixel_value_raises(self):
        """Test an overlay bit within the stored pixel bits raises."""
        ds = make_ds(8, 8)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 11)

        msg = "falls within the stored pixel value bits"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_bit_position_outside_allocated_raises(self):
        """Test an overlay bit outside the allocated bits raises."""
        ds = make_ds(8, 8)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 16)

        msg = "outside the allocated pixel bits"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_overlay_data_present_raises(self):
        """Test an already standalone overlay raises."""
        ds = make_ds(8, 8)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15)
        ds.add_new((0x6000, 0x3000), "OW", b"\x00\x00")

        msg = "'Overlay Data' is already present"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_compressed_raises(self):
        """Test a compressed transfer syntax raises."""
        ds = make_ds(8, 8, tsyntax=JPEGLosslessSV1)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15)

        msg = "compressed transfer syntax"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_too_many_overlay_frames_raises(self):
        """Test overlay frames exceeding the pixel data frames raises."""
        ds = make_ds(8, 8, frames=2)
        ds.PixelData = np.zeros((2, 8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15, frame_origin=2, nr_frames=2)

        msg = "exceed the 2 frame"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_frame_origin_out_of_range_raises(self):
        """Test an out of range Overlay Frame Origin raises."""
        ds = make_ds(8, 8, frames=2)
        ds.PixelData = np.zeros((2, 8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15, frame_origin=3, nr_frames=1)

        msg = "inconsistent with the 2 frame"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_no_overlap_raises(self):
        """Test an overlay plane outside the image raises."""
        ds = make_ds(8, 8)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 4, 4, 15, origin=[20, 1])

        msg = "doesn't overlap the 8 x 8 image"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_samples_per_pixel_raises(self):
        """Test more than 1 sample per pixel raises."""
        ds = make_ds(8, 8)
        ds.SamplesPerPixel = 3
        ds.PixelData = np.zeros((8, 8, 3), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15)

        msg = "1 sample per pixel"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_bits_allocated_mismatch_raises(self):
        """Test Overlay Bits Allocated not matching Bits Allocated raises."""
        ds = make_ds(8, 8)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 7, bits_allocated=8)

        msg = "doesn't match"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_short_pixel_data_raises(self):
        """Test pixel data shorter than expected raises."""
        ds = make_ds(8, 8)
        ds.PixelData = b"\x00" * 10
        add_overlay(ds, 0x6000, 8, 8, 15)

        msg = "less than the expected length"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_missing_element_raises(self):
        """Test a missing required element raises."""
        ds = make_ds(8, 8)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()
        add_overlay(ds, 0x6000, 8, 8, 15)
        del ds[(0x6000, 0x0010)]  # Overlay Rows

        msg = "required elements are missing"
        with pytest.raises(AttributeError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_bad_group_raises(self):
        """Test an invalid overlay group raises."""
        ds = make_ds(8, 8)
        ds.PixelData = np.zeros((8, 8), dtype=np.uint16).tobytes()

        msg = "even number between 0x6000 and 0x60FE"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6001)
