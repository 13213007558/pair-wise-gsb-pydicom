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
        get_overlay_array,
        reshape_overlay_array,
        get_expected_length,
        set_overlay_array,
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

    def test_set_overlay_array_raises(self):
        """Test set_overlay_array raises exception"""
        ds = dcmread(EXPL_1_1_1F)
        msg = r"The overlay data handler requires numpy"
        with pytest.raises(ImportError, match=msg):
            set_overlay_array(ds, 0x6000, None)


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
class TestNumpy_SetOverlayArray:
    """Tests for set_overlay_array writing standalone Overlay Data."""

    def test_bad_group_raises(self):
        """Test that using a bad group raises an exception."""
        ds = dcmread(EXPL_16_1_1F)
        arr = np.zeros((4, 4), dtype="u1")
        msg = (
            r"The group part of the 'Overlay Data' element tag must be "
            r"between 0x6000 and 0x60FF \(inclusive\)"
        )
        with pytest.raises(ValueError, match=msg):
            ds.set_overlay_array(0x5FFF, arr)
        with pytest.raises(ValueError, match=msg):
            ds.set_overlay_array(0x6100, arr)
        with pytest.raises(ValueError, match=msg):
            set_overlay_array(ds, 0x6100, arr)

    def test_non_binary_raises(self):
        """Test that a non-binary array raises an exception."""
        ds = dcmread(EXPL_16_1_1F)
        arr = np.zeros((4, 4), dtype="u1")
        arr[0, 0] = 2
        with pytest.raises(ValueError, match="only contain binary values"):
            ds.set_overlay_array(0x6000, arr)

    def test_bad_shape_raises(self):
        """Test that a badly shaped array raises an exception."""
        ds = dcmread(EXPL_16_1_1F)
        with pytest.raises(ValueError, match="must be shaped"):
            ds.set_overlay_array(0x6000, np.zeros((16,), dtype="u1"))
        with pytest.raises(ValueError, match="must be shaped"):
            ds.set_overlay_array(0x6000, np.zeros((1, 2, 3, 4), dtype="u1"))

    def test_bad_out_of_bounds_value_raises(self):
        """Test that an invalid out_of_bounds value raises an exception."""
        ds = dcmread(EXPL_16_1_1F)
        with pytest.raises(ValueError, match="Invalid 'out_of_bounds' value"):
            ds.set_overlay_array(
                0x6000, np.zeros((4, 4), dtype="u1"), out_of_bounds="shift"
            )

    def test_single_frame(self):
        """Test writing a single frame overlay."""
        ds = dcmread(EXPL_16_1_1F)
        rng = np.random.default_rng(42)
        arr = (rng.random((11, 13)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr)

        assert np.array_equal(ds.overlay_array(0x6000), arr)
        assert ds[0x6000, 0x0010].value == 11  # OverlayRows
        assert ds[0x6000, 0x0011].value == 13  # OverlayColumns
        assert ds[0x6000, 0x0015].value == 1  # NumberOfFramesInOverlay
        assert ds[0x6000, 0x0100].value == 1  # OverlayBitsAllocated
        assert ds[0x6000, 0x0102].value == 0  # OverlayBitPosition
        assert ds[0x6000, 0x0040].value == "R"  # OverlayType
        assert list(ds[0x6000, 0x0050].value) == [1, 1]  # OverlayOrigin
        # 11 * 13 = 143 bits -> 18 bytes, even so no padding byte
        assert len(ds[0x6000, 0x3000].value) == 18

    def test_odd_length_is_padded(self):
        """Test odd length data gets a trailing NULL padding byte."""
        ds = dcmread(EXPL_16_1_1F)
        rng = np.random.default_rng(43)
        # 7 * 7 = 49 bits -> 7 bytes -> padded to 8
        arr = (rng.random((7, 7)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr)

        assert len(ds[0x6000, 0x3000].value) == 8
        assert ds[0x6000, 0x3000].value[-1] == 0
        assert np.array_equal(ds.overlay_array(0x6000), arr)

    def test_row_not_byte_aligned(self):
        """Test rows with a column count that isn't a multiple of 8."""
        ds = dcmread(EXPL_16_1_1F)
        rng = np.random.default_rng(44)
        arr = (rng.random((5, 3)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr)

        # 15 bits -> 2 bytes, packed continuously across rows
        assert len(ds[0x6000, 0x3000].value) == 2
        assert np.array_equal(ds.overlay_array(0x6000), arr)

    def test_multi_frame(self):
        """Test writing a multi-frame overlay."""
        ds = dcmread(EXPL_16_1_1F)
        rng = np.random.default_rng(45)
        arr = (rng.random((3, 9, 7)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr)

        assert ds[0x6000, 0x0015].value == 3  # NumberOfFramesInOverlay
        assert ds[0x6000, 0x0051].value == 1  # ImageFrameOrigin
        read_back = ds.overlay_array(0x6000)
        assert read_back.shape == (3, 9, 7)
        assert np.array_equal(read_back, arr)

    def test_bool_array(self):
        """Test writing a bool dtype array."""
        ds = dcmread(EXPL_16_1_1F)
        arr = np.zeros((8, 8), dtype=bool)
        arr[::2, ::2] = True
        ds.set_overlay_array(0x6000, arr)

        assert np.array_equal(ds.overlay_array(0x6000), arr.astype("u1"))

    def test_replaces_existing_overlay(self):
        """Test writing over an existing overlay."""
        ds = dcmread(EXPL_16_1_1F)
        ds.set_overlay_array(0x6000, np.ones((484, 484), dtype="u1"))
        rng = np.random.default_rng(46)
        arr = (rng.random((16, 16)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr)

        assert np.array_equal(ds.overlay_array(0x6000), arr)
        assert ds[0x6000, 0x0010].value == 16
        assert ds[0x6000, 0x0011].value == 16

    def test_roundtrip_file(self, tmp_path):
        """Test the written overlay survives a save/read round-trip."""
        ds = dcmread(EXPL_16_1_1F)
        rng = np.random.default_rng(47)
        arr = (rng.random((7, 7)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr)

        fpath = tmp_path / "overlay.dcm"
        ds.save_as(fpath, enforce_file_format=True)
        ds2 = dcmread(fpath)
        assert np.array_equal(ds2.overlay_array(0x6000), arr)


def _make_image_ds(rows=16, columns=16, frames=1, bits_allocated=16, bits_stored=12):
    """Return a dataset with uncompressed pixel data and its pixel array."""
    ds = Dataset()
    ds.SOPClassUID = "1.2.3.4"
    ds.SOPInstanceUID = "1.2.3.4.5"
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_stored
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    if frames > 1:
        ds.NumberOfFrames = frames

    rng = np.random.default_rng(1234)
    dtype = "<u2" if bits_allocated == 16 else "u1"
    arr = rng.integers(0, 1 << bits_stored, size=(frames, rows, columns)).astype(dtype)
    ds.PixelData = arr.tobytes()
    return ds, arr


def _pixel_array(ds, frames, rows, columns, bits_allocated=16):
    """Return the dataset's pixel data as an array."""
    dtype = "<u2" if bits_allocated == 16 else "u1"
    return np.frombuffer(ds.PixelData, dtype=dtype).reshape(frames, rows, columns)


@pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")
class TestNumpy_SetEmbeddedOverlay:
    """Tests for set_overlay_array with embedded=True."""

    def test_embedded_basic(self):
        """Test writing an embedded overlay bit plane."""
        ds, original = _make_image_ds()
        rng = np.random.default_rng(50)
        arr = (rng.random((16, 16)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

        px = _pixel_array(ds, 1, 16, 16)
        # The bit plane holds the overlay
        assert np.array_equal((px[0] >> 15) & 1, arr)
        # All other bits are unchanged
        assert np.array_equal(px & 0x7FFF, original & 0x7FFF)
        # The overlay elements have been synchronised
        assert ds[0x6000, 0x0100].value == 16  # OverlayBitsAllocated
        assert ds[0x6000, 0x0102].value == 15  # OverlayBitPosition
        assert ds[0x6000, 0x0010].value == 16  # OverlayRows
        assert ds[0x6000, 0x0011].value == 16  # OverlayColumns
        assert ds[0x6000, 0x0015].value == 1  # NumberOfFramesInOverlay
        # Embedded overlays have no standalone Overlay Data element
        assert (0x6000, 0x3000) not in ds

    def test_embedded_removes_overlay_data(self):
        """Test an existing Overlay Data element is removed."""
        ds, _ = _make_image_ds()
        ds.add_new((0x6000, 0x3000), "OW", b"\x00\x00")
        arr = np.ones((16, 16), dtype="u1")
        ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)
        assert (0x6000, 0x3000) not in ds

    def test_embedded_bit_position_from_element(self):
        """Test the bit position is taken from the dataset if not given."""
        ds, original = _make_image_ds()
        ds.add_new((0x6000, 0x0102), "US", 14)  # OverlayBitPosition
        arr = np.ones((16, 16), dtype="u1")
        ds.set_overlay_array(0x6000, arr, embedded=True)

        px = _pixel_array(ds, 1, 16, 16)
        assert (px[0] >> 14 & 1).all()
        assert np.array_equal(px & 0xBFFF, original & 0xBFFF)
        assert ds[0x6000, 0x0102].value == 14

    def test_embedded_no_bit_position_raises(self):
        """Test missing bit position raises an exception."""
        ds, _ = _make_image_ds()
        arr = np.ones((16, 16), dtype="u1")
        with pytest.raises(ValueError, match="no 'bit_position' was given"):
            ds.set_overlay_array(0x6000, arr, embedded=True)

    def test_embedded_invalid_bit_position_raises(self):
        """Test an out of range bit position raises an exception."""
        ds, _ = _make_image_ds()
        arr = np.ones((16, 16), dtype="u1")
        with pytest.raises(ValueError, match="bit position 16 is invalid"):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=16)

    def test_embedded_overwrites_stored_bits_raises(self):
        """Test a bit plane overlapping the used pixel bits is rejected."""
        ds, _ = _make_image_ds(bits_stored=12)
        original_px = ds.PixelData
        arr = np.ones((16, 16), dtype="u1")
        msg = r"writing would overwrite valid pixel bits"
        with pytest.raises(ValueError, match=msg):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=11)
        assert ds.PixelData == original_px

    def test_embedded_compressed_raises(self):
        """Test compressed pixel data is rejected."""
        ds, _ = _make_image_ds()
        ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
        arr = np.ones((16, 16), dtype="u1")
        with pytest.raises(ValueError, match="encapsulated .compressed"):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

    def test_embedded_no_pixel_data_raises(self):
        """Test a dataset without Pixel Data raises an exception."""
        ds, _ = _make_image_ds()
        del ds["PixelData"]
        arr = np.ones((16, 16), dtype="u1")
        with pytest.raises(ValueError, match=r"no \(7FE0,0010\)"):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

    def test_embedded_missing_image_element_raises(self):
        """Test a missing image element raises an exception."""
        ds, _ = _make_image_ds()
        del ds["BitsStored"]
        arr = np.ones((16, 16), dtype="u1")
        with pytest.raises(AttributeError, match="BitsStored"):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

    def test_embedded_samples_per_pixel_raises(self):
        """Test SamplesPerPixel > 1 raises an exception."""
        ds, _ = _make_image_ds()
        ds.SamplesPerPixel = 3
        arr = np.ones((16, 16), dtype="u1")
        with pytest.raises(ValueError, match="Samples per Pixel"):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

    def test_embedded_8_bit(self):
        """Test writing an embedded overlay to 8-bit pixel data."""
        ds, original = _make_image_ds(bits_allocated=8, bits_stored=6)
        rng = np.random.default_rng(51)
        arr = (rng.random((16, 16)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=7)

        px = _pixel_array(ds, 1, 16, 16, bits_allocated=8)
        assert np.array_equal((px[0] >> 7) & 1, arr)
        assert np.array_equal(px & 0x7F, original & 0x7F)
        assert ds[0x6000, 0x0100].value == 8

    def test_embedded_origin_mapping(self):
        """Test the overlay is mapped using OverlayOrigin."""
        ds, original = _make_image_ds()
        ds.add_new((0x6000, 0x0050), "SS", [3, 5])  # OverlayOrigin
        arr = np.ones((4, 6), dtype="u1")
        ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

        px = _pixel_array(ds, 1, 16, 16)
        plane = (px[0] >> 15) & 1
        # OverlayOrigin is 1-based: rows 3..6, columns 5..10
        assert plane[2:6, 4:10].all()
        assert plane.sum() == 24
        assert np.array_equal(px & 0x7FFF, original & 0x7FFF)

    def test_embedded_frame_mapping(self):
        """Test the overlay frames are mapped using ImageFrameOrigin."""
        ds, original = _make_image_ds(frames=3)
        ds.add_new((0x6000, 0x0051), "US", 2)  # ImageFrameOrigin
        rng = np.random.default_rng(52)
        arr = (rng.random((2, 16, 16)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

        px = _pixel_array(ds, 3, 16, 16)
        # Overlay frame 0 -> image frame 2 (1-based), etc.
        assert not (px[0] >> 15).any()
        assert np.array_equal((px[1] >> 15) & 1, arr[0])
        assert np.array_equal((px[2] >> 15) & 1, arr[1])
        assert np.array_equal(px & 0x7FFF, original & 0x7FFF)
        assert ds[0x6000, 0x0015].value == 2  # NumberOfFramesInOverlay

    def test_embedded_out_of_bounds_raises(self):
        """Test an out of bounds overlay raises by default."""
        ds, _ = _make_image_ds()
        original_px = ds.PixelData
        ds.add_new((0x6000, 0x0050), "SS", [15, 15])  # OverlayOrigin
        arr = np.ones((4, 4), dtype="u1")
        with pytest.raises(ValueError, match="extends outside the image"):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)
        assert ds.PixelData == original_px

    def test_embedded_frame_out_of_bounds_raises(self):
        """Test out of bounds overlay frames raise by default."""
        ds, _ = _make_image_ds(frames=3)
        ds.add_new((0x6000, 0x0051), "US", 3)  # ImageFrameOrigin
        arr = np.ones((2, 16, 16), dtype="u1")
        with pytest.raises(ValueError, match="extends outside the image"):
            ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

    def test_embedded_out_of_bounds_crop(self):
        """Test cropping an out of bounds overlay to the image."""
        ds, original = _make_image_ds()
        ds.add_new((0x6000, 0x0050), "SS", [15, 15])  # OverlayOrigin
        arr = np.ones((4, 4), dtype="u1")
        ds.set_overlay_array(
            0x6000, arr, embedded=True, bit_position=15, out_of_bounds="crop"
        )

        px = _pixel_array(ds, 1, 16, 16)
        plane = (px[0] >> 15) & 1
        # Only the in-bounds 2x2 region at rows/cols 15..16 (1-based) is set
        assert plane.sum() == 4
        assert plane[14:16, 14:16].all()
        assert np.array_equal(px & 0x7FFF, original & 0x7FFF)

    def test_embedded_entirely_outside_raises(self):
        """Test an overlay entirely outside the image raises when cropping."""
        ds, _ = _make_image_ds()
        ds.add_new((0x6000, 0x0050), "SS", [100, 100])  # OverlayOrigin
        arr = np.ones((4, 4), dtype="u1")
        with pytest.raises(ValueError, match="lies entirely outside"):
            ds.set_overlay_array(
                0x6000,
                arr,
                embedded=True,
                bit_position=15,
                out_of_bounds="crop",
            )

    def test_embedded_big_endian(self):
        """Test writing an embedded overlay to big endian pixel data."""
        ds, original = _make_image_ds()
        ds.file_meta.TransferSyntaxUID = ExplicitVRBigEndian
        ds.PixelData = original.astype(">u2").tobytes()
        rng = np.random.default_rng(53)
        arr = (rng.random((16, 16)) > 0.5).astype("u1")
        ds.set_overlay_array(0x6000, arr, embedded=True, bit_position=15)

        px = np.frombuffer(ds.PixelData, dtype=">u2").reshape(1, 16, 16)
        assert np.array_equal((px[0] >> 15) & 1, arr)
        assert np.array_equal(px & 0x7FFF, original & 0x7FFF)
