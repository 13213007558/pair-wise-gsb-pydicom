# Copyright 2008-2019 pydicom authors. See LICENSE file for details.
"""Tests for the overlays.numpy_handler module.

There are the following possibilities:

* numpy is not available and
  * the numpy handler is not available
  * the numpy handler is available
* numpy is available and
  * the numpy handler is not available
  * the numpy handler is available

**Supported transfer syntaxes**

* 1.2.840.10008.1.2 : Implicit VR Little Endian
* 1.2.840.10008.1.2.1 : Explicit VR Little Endian
* 1.2.840.10008.1.2.1.99 : Deflated Explicit VR Little Endian
* 1.2.840.10008.1.2.2 : Explicit VR Big Endian

**Elements affecting the handler**

* OverlayBitsAllocated
* NumberOfFramesInOverlay
* OverlayRows
* OverlayColumns
"""

import pytest

import pydicom
from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filereader import dcmread
from pydicom.pixels.utils import pack_bits, unpack_bits
from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    JPEGBaseline8Bit,
)

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

try:
    from pydicom.overlays import numpy_handler as NP_HANDLER
    from pydicom.overlays.numpy_handler import (
        extract_overlay,
        get_overlay_array,
        reshape_overlay_array,
        get_expected_length,
    )
except ImportError:
    NP_HANDLER = None


# Paths to the test datasets
# EXPL: Explicit VR Little Endian
# Overlay Data
# 1/1, 1 sample/pixel, 1 frame
EXPL_1_1_1F = get_testdata_file("MR-SIEMENS-DICOM-WithOverlays.dcm")
# 1/1, 1 sample/pixel, N frame
EXPL_1_1_3F = None
# No Overlay Data
# 16/16, 1 sample/pixel, 1 frame
EXPL_16_1_1F = get_testdata_file("MR_small.dcm")


# Numpy unavailable and the numpy handler is available
@pytest.mark.skipif(HAVE_NP, reason="Numpy is available")
class TestNoNumpy_NumpyHandler:
    """Tests for handling datasets without numpy and the handler."""

    def test_get_overlay_array_raises(self):
        """Test get_overlay_array raises exception"""
        ds = dcmread(EXPL_1_1_1F)
        msg = r"The overlay data handler requires numpy"
        with pytest.raises(ImportError, match=msg):
            get_overlay_array(ds, 0x6000)


@pytest.mark.skipif(HAVE_NP, reason="Numpy is available")
def test_reshape_pixel_array_raises():
    """Test that reshape_overlay_array raises exception without numpy."""
    with pytest.raises(ImportError, match="Numpy is required to reshape"):
        reshape_overlay_array(None, None)


# Numpy and the numpy handler are available
EXPL = ExplicitVRLittleEndian
IMPL = ImplicitVRLittleEndian
REFERENCE_DATA_LITTLE = [
    # fpath, (syntax, bits, nr samples, pixel repr, nr frames, shape, dtype,
    #   group)
    (EXPL_1_1_1F, (EXPL, 1, 1, 0, 1, (484, 484), "uint8", 0x6000)),
    # (EXPL_1_1_3F, (EXPL, 1, 1, 0, 3, (3, 512, 512), 'uint8', 0x6000)),
]


@pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")
class TestNumpy_NumpyHandler:
    """Tests for handling Overlay Data with the handler."""

    # Little endian datasets
    @pytest.mark.parametrize("fpath, data", REFERENCE_DATA_LITTLE)
    def test_properties(self, fpath, data):
        """Test dataset and overlay array properties are as expected."""
        ds = dcmread(fpath)
        group = data[7]
        assert ds.file_meta.TransferSyntaxUID == data[0]
        assert ds[group, 0x0100].value == data[1]  # OverlayBitsAllocated

        arr = ds.overlay_array(data[7])
        assert data[5] == arr.shape
        assert arr.dtype == data[6]

        # Odd sized data is padded by a final 0x00 byte
        rows = ds[group, 0x0010].value
        columns = ds[group, 0x0011].value
        nr_frames = ds[group, 0x0015].value
        size = rows * columns * nr_frames / 8 * data[2]
        assert len(ds[group, 0x3000].value) == size + size % 2
        if size % 2:
            assert ds[group, 0x3000].value[-1] == b"\x00"[0]

    def test_little_1bit_1sample_1frame(self):
        """Test pixel_array for little 1-bit, 1 sample/pixel, 1 frame."""
        ds = dcmread(EXPL_1_1_1F)
        arr = ds.overlay_array(0x6000)

        assert arr.flags.writeable
        assert arr.max() == 1
        assert arr.min() == 0
        assert 29 == sum(arr[422, 393:422])

    def test_read_only(self):
        """Test for #717, returned array read-only."""
        ds = dcmread(EXPL_1_1_1F)
        arr = ds.overlay_array(0x6000)
        assert 0 == arr[0, 0]
        arr[0, 0] = 1
        assert 1 == arr[0, 0]
        assert arr.flags.writeable

    def test_bad_group_raises(self):
        """Test that using a bad group raises exception."""
        ds = dcmread(EXPL_1_1_1F)
        msg = (
            r"The group part of the 'Overlay Data' element tag must be "
            r"between 0x6000 and 0x60FF \(inclusive\)"
        )
        with pytest.raises(ValueError, match=msg):
            ds.overlay_array(0x5FFF)
        with pytest.raises(ValueError, match=msg):
            ds.overlay_array(0x6100)

    def test_no_frames(self):
        """Test handler with no NumberOfFramesInOverlay element."""
        ds = dcmread(EXPL_1_1_1F)
        del ds[0x6000, 0x0015]
        arr = ds.overlay_array(0x6000)

        assert arr.max() == 1
        assert arr.min() == 0
        assert 29 == sum(arr[422, 393:422])


# Tests for numpy_handler module with Numpy available
@pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")
class TestNumpy_GetOverlayArray:
    """Tests for numpy_handler.get_overlay_array with numpy."""

    def test_no_overlay_data_raises(self):
        """Test get_overlay_array raises if dataset has no OverlayData."""
        ds = dcmread(EXPL_1_1_1F)
        del ds[0x6000, 0x3000]
        assert (0x6000, 0x3000) not in ds
        with pytest.raises(AttributeError, match=r" dataset: OverlayData"):
            get_overlay_array(ds, 0x6000)

    def test_bad_length_raises(self):
        """Test bad pixel data length raises exception."""
        ds = dcmread(EXPL_1_1_1F)
        # Too short
        ds[0x6000, 0x3000].value = ds[0x6000, 0x3000][:-1]
        msg = (
            r"The length of the overlay data in the dataset \(29281 bytes\) "
            r"doesn't match the expected length \(29282 bytes\). "
            r"The dataset may be corrupted or there may be an issue "
            r"with the overlay data handler."
        )
        with pytest.raises(ValueError, match=msg):
            get_overlay_array(ds, 0x6000)

    def test_missing_padding_warns(self):
        """A warning shall be issued if the padding for odd data is missing."""
        ds = dcmread(EXPL_1_1_1F)
        # Edit shape
        ds[0x6000, 0x0010].value = 15  # OverlayRows
        ds[0x6000, 0x0011].value = 14  # OverlayColumns
        ds[0x6000, 0x3000].value = ds[0x6000, 0x3000].value[:27]  # 15 * 14 / 8
        msg = r"The overlay data length is odd and misses a padding byte."
        with pytest.warns(UserWarning, match=msg):
            get_overlay_array(ds, 0x6000)

    def test_excess_padding(self):
        """A warning shall be issued excess padding present."""
        ds = dcmread(EXPL_1_1_1F)
        # Edit shape
        ds[0x6000, 0x0010].value = 15  # OverlayRows
        ds[0x6000, 0x0011].value = 14  # OverlayColumns
        overlay_data = ds[0x6000, 0x3000].value[:27] + b"\x00\x00\x00"
        ds[0x6000, 0x3000].value = overlay_data
        msg = (
            r"overlay data in the dataset \(30 bytes\) indicates it contains "
            r"excess padding. 3 bytes will be removed"
        )
        with pytest.warns(UserWarning, match=msg):
            get_overlay_array(ds, 0x6000)


if HAVE_NP:
    RESHAPE_ARRAYS = {
        "reference": np.asarray(
            [
                [  # Frame 1
                    [[1, 9, 17], [2, 10, 18], [3, 11, 19], [4, 12, 20], [5, 13, 21]],
                    [[2, 10, 18], [3, 11, 19], [4, 12, 20], [5, 13, 21], [6, 14, 22]],
                    [[3, 11, 19], [4, 12, 20], [5, 13, 21], [6, 14, 22], [7, 15, 23]],
                    [[4, 12, 20], [5, 13, 21], [6, 14, 22], [7, 15, 23], [8, 16, 24]],
                ],
                [  # Frame 2
                    [
                        [25, 33, 41],
                        [26, 34, 42],
                        [27, 35, 43],
                        [28, 36, 44],
                        [29, 37, 45],
                    ],
                    [
                        [26, 34, 42],
                        [27, 35, 43],
                        [28, 36, 44],
                        [29, 37, 45],
                        [30, 38, 46],
                    ],
                    [
                        [27, 35, 43],
                        [28, 36, 44],
                        [29, 37, 45],
                        [30, 38, 46],
                        [31, 39, 47],
                    ],
                    [
                        [28, 36, 44],
                        [29, 37, 45],
                        [30, 38, 46],
                        [31, 39, 47],
                        [32, 40, 48],
                    ],
                ],
            ]
        ),
        "1frame_1sample": np.asarray(
            [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8]
        ),
        "2frame_1sample": np.asarray(
            [
                1,
                2,
                3,
                4,
                5,
                2,
                3,
                4,
                5,
                6,  # Frame 1
                3,
                4,
                5,
                6,
                7,
                4,
                5,
                6,
                7,
                8,
                25,
                26,
                27,
                28,
                29,
                26,
                27,
                28,
                29,
                30,  # Frame 2
                27,
                28,
                29,
                30,
                31,
                28,
                29,
                30,
                31,
                32,
            ]
        ),
    }


@pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")
class TestNumpy_ReshapeOverlayArray:
    """Tests for numpy_handler.reshape_overlay_array."""

    def setup_method(self):
        """Setup the test dataset."""
        self.elem = {
            "OverlayRows": 4,
            "OverlayColumns": 5,
        }

        # Expected output ref_#frames_#samples
        self.ref_1_1 = RESHAPE_ARRAYS["reference"][0, :, :, 0]
        self.ref_2_1 = RESHAPE_ARRAYS["reference"][:, :, :, 0]

    def test_reference_1frame_1sample(self):
        """Test the 1 frame 1 sample/pixel reference array is as expected."""
        # (rows, columns)
        assert (4, 5) == self.ref_1_1.shape
        assert np.array_equal(
            self.ref_1_1,
            np.asarray(
                [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]
            ),
        )

    def test_reference_2frame_1sample(self):
        """Test the 2 frame 1 sample/pixel reference array is as expected."""
        # (nr frames, rows, columns)
        assert (2, 4, 5) == self.ref_2_1.shape

        # Frame 1
        assert np.array_equal(
            self.ref_2_1[0, :, :],
            np.asarray(
                [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]
            ),
        )
        # Frame 2
        assert np.array_equal(
            self.ref_2_1[1, :, :],
            np.asarray(
                [
                    [25, 26, 27, 28, 29],
                    [26, 27, 28, 29, 30],
                    [27, 28, 29, 30, 31],
                    [28, 29, 30, 31, 32],
                ]
            ),
        )

    def test_1frame(self):
        """Test reshaping 1 frame, 1 sample/pixel."""
        self.elem["NumberOfFramesInOverlay"] = 1
        arr = reshape_overlay_array(self.elem, RESHAPE_ARRAYS["1frame_1sample"])
        assert (4, 5) == arr.shape
        assert np.array_equal(arr, self.ref_1_1)

    def test_2frame_1sample(self):
        """Test reshaping 2 frame, 1 sample/pixel."""
        self.elem["NumberOfFramesInOverlay"] = 2
        arr = reshape_overlay_array(self.elem, RESHAPE_ARRAYS["2frame_1sample"])
        assert (2, 4, 5) == arr.shape
        assert np.array_equal(arr, self.ref_2_1)

    def test_invalid_nr_frames_raises(self):
        """Test an invalid Number of Frames value raises exception."""
        self.elem["NumberOfFramesInOverlay"] = 0
        # Need to escape brackets
        with pytest.raises(ValueError, match=r"value of 0 for \(60xx,0015\)"):
            reshape_overlay_array(self.elem, RESHAPE_ARRAYS["1frame_1sample"])


REFERENCE_LENGTH = [
    # (frames, rows, cols), bit depth, result in (bytes, pixels)
    ((1, 0, 0), 1, (0, 0)),
    ((1, 1, 1), 1, (1, 1)),  # 1 bit -> 1 byte
    ((1, 2, 2), 1, (1, 4)),  # 4 bits -> 1 byte
    ((1, 2, 4), 1, (1, 8)),  # 8 bits -> 1 byte
    ((1, 3, 3), 1, (2, 9)),  # 9 bits -> 2 bytes
    ((1, 512, 512), 1, (32768, 262144)),  # Typical length
    # NumberOfFramesInOverlay odd and > 1
    ((3, 0, 0), 1, (0, 0)),
    ((3, 1, 1), 1, (1, 3)),
    ((3, 2, 4), 1, (3, 24)),
    ((3, 2, 2), 1, (2, 12)),
    ((3, 3, 3), 1, (4, 27)),
    ((3, 512, 512), 1, (98304, 786432)),
    # NumberOfFramesInOverlay even
    ((4, 0, 0), 1, (0, 0)),
    ((4, 1, 1), 1, (1, 4)),
    ((4, 2, 4), 1, (4, 32)),
    ((4, 2, 2), 1, (2, 16)),
    ((4, 3, 3), 1, (5, 36)),
    ((4, 512, 512), 1, (131072, 1048576)),
]


@pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")
class TestNumpy_GetExpectedLength:
    """Tests for numpy_handler.get_expected_length."""

    @pytest.mark.parametrize("shape, bits, length", REFERENCE_LENGTH)
    def test_length_in_bytes(self, shape, bits, length):
        """Test get_expected_length(ds, unit='bytes')."""
        elem = {
            "OverlayRows": shape[1],
            "OverlayColumns": shape[2],
            "OverlayBitsAllocated": bits,
            "NumberOfFramesInOverlay": shape[0],
        }

        assert length[0] == get_expected_length(elem, unit="bytes")

    @pytest.mark.parametrize("shape, bits, length", REFERENCE_LENGTH)
    def test_length_in_pixels(self, shape, bits, length):
        """Test get_expected_length(ds, unit='pixels')."""
        elem = {
            "OverlayRows": shape[1],
            "OverlayColumns": shape[2],
            "OverlayBitsAllocated": bits,
            "NumberOfFramesInOverlay": shape[0],
        }

        assert length[1] == get_expected_length(elem, unit="pixels")


@pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")
class TestExtractOverlay:
    """Tests for numpy_handler.extract_overlay."""

    def make_ds(
        self,
        plane,
        rows=6,
        columns=6,
        frames=1,
        bits_allocated=16,
        bits_stored=12,
        pixel_representation=0,
        tsyntax=EXPL,
        group=0x6000,
        origin=(1, 1),
        bit_position=15,
        frame_origin=1,
        other_bits=(),
    ):
        """Return a dataset with the overlay plane embedded in the pixel
        data, along with the pixel cells before and the pixel cells expected
        after extraction.
        """
        plane = np.asarray(plane, dtype="u1")
        if plane.ndim == 2:
            plane = plane[np.newaxis, ...]
        overlay_frames, overlay_rows, overlay_columns = plane.shape

        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = tsyntax
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = pixel_representation
        ds.BitsAllocated = bits_allocated
        ds.BitsStored = bits_stored
        ds.HighBit = bits_stored - 1
        ds.Rows = rows
        ds.Columns = columns
        if frames > 1:
            ds.NumberOfFrames = str(frames)

        # Deterministic pixel values within the stored bits
        index = np.arange(frames * rows * columns)
        arr = (index * 37 % 2**bits_stored).astype(f"uint{bits_allocated}")
        arr = arr.reshape(frames, rows, columns)
        # A pattern in bits used by other (non-extracted) overlays
        for bit in other_bits:
            arr |= ((index * 13 % 2) << bit).astype(arr.dtype).reshape(arr.shape)

        # Embed the overlay plane, clipped to the image bounds
        r0, c0 = origin[1] - 1, origin[0] - 1
        rs, re = max(r0, 0), min(r0 + overlay_rows, rows)
        cs, ce = max(c0, 0), min(c0 + overlay_columns, columns)
        first = frame_origin - 1
        if bit_position < bits_allocated:
            for f in range(min(overlay_frames, frames - first)):
                region = plane[f, rs - r0 : re - r0, cs - c0 : ce - c0]
                arr[first + f, rs:re, cs:ce] |= region.astype(arr.dtype) << bit_position

        byteorder = "<" if tsyntax.is_little_endian else ">"
        ds.PixelData = arr.astype(f"{byteorder}u{bits_allocated // 8}").tobytes()

        ds.add_new((group, 0x0010), "US", overlay_rows)
        ds.add_new((group, 0x0011), "US", overlay_columns)
        ds.add_new((group, 0x0050), "SS", list(origin))
        ds.add_new((group, 0x0100), "US", bits_allocated)
        ds.add_new((group, 0x0102), "US", bit_position)
        if overlay_frames > 1:
            ds.add_new((group, 0x0015), "IS", overlay_frames)
        if frame_origin != 1:
            ds.add_new((group, 0x0051), "US", frame_origin)

        # The expected pixel cells after the overlay's bits are cleared
        after = arr.copy()
        if bit_position < bits_allocated:
            mask = arr.dtype.type(1 << bit_position)
            after[first : first + overlay_frames, rs:re, cs:ce] &= ~mask

        return ds, arr, after, plane

    def test_extract_basic(self):
        """Test extracting an overlay embedded in the pixel data."""
        plane = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 1],
            ]
        )
        ds, before, after, plane = self.make_ds(plane, origin=(2, 3), other_bits=(14,))
        original_pixel_data = ds.PixelData

        out = extract_overlay(ds, 0x6000)
        assert out is ds

        # The standalone overlay elements have been added/updated
        assert ds[0x6000, 0x0100].value == 1
        assert ds[0x6000, 0x0102].value == 0
        # 15 overlay bits, packed LSB-first into 2 bytes
        assert ds[0x6000, 0x3000].value == b"\x55\x4d"
        assert np.array_equal(get_overlay_array(ds, 0x6000), plane[0])

        # Only the overlay's bits in its region have been cleared
        assert ds.PixelData == after.astype("<u2").tobytes()
        # The other overlay's bits are unchanged
        assert np.array_equal(
            np.frombuffer(ds.PixelData, dtype="<u2") & (1 << 14),
            np.frombuffer(original_pixel_data, dtype="<u2") & (1 << 14),
        )
        # The stored pixel values are unchanged
        assert np.array_equal(ds.pixel_array, (before & 0x0FFF).reshape(6, 6))

    def test_odd_width_packing(self):
        """Test packing when the overlay width is not a multiple of 8."""
        plane = np.ones((2, 7), dtype="u1")
        ds, _, _, _ = self.make_ds(plane, rows=2, columns=7)
        extract_overlay(ds, 0x6000)
        # 14 bits -> 2 bytes, already even so no padding byte
        assert ds[0x6000, 0x3000].value == b"\xff\x3f"
        assert np.array_equal(get_overlay_array(ds, 0x6000), plane)

    def test_negative_origin(self):
        """Test an overlay with a negative origin is clipped to the image."""
        plane = np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 1, 1],
            ]
        )
        ds, before, after, plane = self.make_ds(
            plane, rows=4, columns=4, origin=(-1, 0)
        )
        # Set the overlay bit outside the overlay's region
        arr = np.frombuffer(ds.PixelData, dtype="<u2").reshape(4, 4).copy()
        arr[3, 3] |= 1 << 15
        ds.PixelData = arr.tobytes()

        extract_overlay(ds, 0x6000)

        # Overlay pixels outside the image are 0, the rest match the plane
        expected_plane = np.zeros((3, 4), dtype="u1")
        expected_plane[1:3, 2:4] = plane[0, 1:3, 2:4]
        assert np.array_equal(get_overlay_array(ds, 0x6000), expected_plane)

        out = np.frombuffer(ds.PixelData, dtype="<u2").reshape(4, 4)
        # The overlay's bits are cleared only within its region
        assert not (out[0:2, 0:2] & (1 << 15)).any()
        # Bits outside the overlay's region are unchanged
        assert out[3, 3] & (1 << 15)
        assert np.array_equal(out & 0x0FFF, arr & 0x0FFF)

    def test_multiframe(self):
        """Test multi-frame overlays are packed continuously across frames."""
        plane = np.array(
            [
                [[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]],
                [[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]],
            ]
        )
        ds, before, after, plane = self.make_ds(
            plane, rows=4, columns=3, frames=3, frame_origin=2
        )
        # Set the overlay bit in an image frame the overlay doesn't apply to
        arr = np.frombuffer(ds.PixelData, dtype="<u2").reshape(3, 4, 3).copy()
        arr[0] |= 1 << 15
        ds.PixelData = arr.tobytes()

        extract_overlay(ds, 0x6000)

        # 24 bits packed continuously across the frame boundary
        assert ds[0x6000, 0x3000].value == pack_bits(plane.ravel())
        assert np.array_equal(get_overlay_array(ds, 0x6000), plane)

        out = np.frombuffer(ds.PixelData, dtype="<u2").reshape(3, 4, 3)
        # The frame the overlay doesn't apply to is unchanged
        assert np.array_equal(out[0], arr[0])
        # The overlay's bits are cleared from the frames it applies to
        assert not (out[1:] & (1 << 15)).any()
        assert np.array_equal(out & 0x0FFF, arr & 0x0FFF)

    def test_signed_pixels(self):
        """Test extraction with signed pixel representation."""
        plane = np.array([[1, 1, 0], [0, 1, 1]])
        ds, before, after, _ = self.make_ds(
            plane, rows=2, columns=3, pixel_representation=1
        )
        extract_overlay(ds, 0x6000)
        assert np.array_equal(ds.pixel_array, after.reshape(2, 3))
        assert np.array_equal(ds.pixel_array, (before & 0x0FFF).reshape(2, 3))

    def test_big_endian(self):
        """Test extraction with a big endian transfer syntax."""
        plane = np.array([[1, 0, 1], [0, 1, 0]])
        ds, before, after, _ = self.make_ds(
            plane, rows=2, columns=3, tsyntax=ExplicitVRBigEndian
        )
        extract_overlay(ds, 0x6000)
        assert ds.PixelData == after.astype(">u2").tobytes()
        assert np.array_equal(get_overlay_array(ds, 0x6000), plane)

    def test_raises_invalid_group(self):
        """Test an invalid overlay group raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"))
        msg = "The overlay group must be an even number"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6001)

    def test_raises_missing_overlay_element(self):
        """Test a missing overlay element raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"))
        del ds[0x6000, 0x0010]
        msg = "required elements are missing.*OverlayRows"
        with pytest.raises(AttributeError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_missing_pixel_data(self):
        """Test missing pixel data raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"))
        del ds.PixelData
        msg = "required elements are missing.*PixelData"
        with pytest.raises(AttributeError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_already_standalone(self):
        """Test an overlay not stored in the pixel data raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"))
        ds[0x6000, 0x0100].value = 1
        msg = "not stored in the pixel data"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_overlay_data_present(self):
        """Test existing overlay data raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"))
        ds.add_new((0x6000, 0x3000), "OW", b"\x00\x00")
        msg = "'Overlay Data' is already present"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_compressed(self):
        """Test a compressed transfer syntax raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"), tsyntax=JPEGBaseline8Bit)
        msg = "compressed transfer syntax"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_bit_in_stored_bits(self):
        """Test an overlay bit within the stored bits raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"), bit_position=11)
        msg = "falls within the stored pixel bits"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_bit_outside_allocated(self):
        """Test an overlay bit outside the allocated bits raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"), bit_position=16)
        msg = "outside the allocated bits"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_mismatched_bits_allocated(self):
        """Test mismatched bits allocated raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"))
        ds[0x6000, 0x0100].value = 8
        msg = "doesn't match"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_frames_out_of_range(self):
        """Test overlay frames exceeding the image frames raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2, 2), dtype="u1"), frames=1)
        msg = "exceed the number of image frames"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)

    def test_raises_overlay_outside_image(self):
        """Test an overlay region outside the image raises an exception."""
        ds, *_ = self.make_ds(np.ones((2, 2), dtype="u1"), origin=(10, 10))
        msg = "lies outside the 6 x 6 pixel data"
        with pytest.raises(ValueError, match=msg):
            extract_overlay(ds, 0x6000)
