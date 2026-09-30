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
from pydicom.filereader import dcmread
from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    JPEG2000,
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
    """Tests for numpy_handler.set_overlay_array standalone Overlay Data."""

    def test_standalone_roundtrip(self):
        """A single-frame mask shall round-trip with synced elements."""
        ds = pydicom.Dataset()
        # 10 x 10 = 100 bits -> 13 bytes, padded to an even 14 bytes
        mask = (np.arange(100).reshape(10, 10) % 3 == 0).astype("u1")
        set_overlay_array(ds, 0x6000, mask)

        assert np.array_equal(ds.overlay_array(0x6000), mask)
        assert len(ds[0x6000, 0x3000].value) == 14
        assert ds[0x6000, 0x3000].value[-1] == 0
        assert ds[0x6000, 0x0010].value == 10  # OverlayRows
        assert ds[0x6000, 0x0011].value == 10  # OverlayColumns
        assert (0x6000, 0x0015) not in ds  # single frame: no frame count
        assert ds[0x6000, 0x0100].value == 1  # OverlayBitsAllocated
        assert ds[0x6000, 0x0102].value == 0  # OverlayBitPosition
        assert list(ds[0x6000, 0x0050].value) == [1, 1]  # OverlayOrigin
        assert ds[0x6000, 0x0051].value == 1  # ImageFrameOrigin

    def test_standalone_multiframe_roundtrip(self):
        """A multi-frame mask shall round-trip with frame elements."""
        ds = pydicom.Dataset()
        mask = np.zeros((3, 7, 5), dtype="u1")
        mask[1, 2:5, 1:4] = 1
        set_overlay_array(
            ds, 0x6012, mask, origin=(10, 20), image_frame_origin=4
        )

        assert np.array_equal(ds.overlay_array(0x6012), mask)
        assert ds[0x6012, 0x0015].value == 3
        assert list(ds[0x6012, 0x0050].value) == [10, 20]
        assert ds[0x6012, 0x0051].value == 4

    def test_standalone_replaces_embedded_elements(self):
        """Writing standalone overlay data resets the embedded elements."""
        ds = pydicom.Dataset()
        ds.add_new((0x6000, 0x0100), "US", 16)
        ds.add_new((0x6000, 0x0102), "US", 15)
        mask = np.ones((4, 6), dtype="u1")
        set_overlay_array(ds, 0x6000, mask)

        assert np.array_equal(ds.overlay_array(0x6000), mask)
        assert ds[0x6000, 0x0100].value == 1
        assert ds[0x6000, 0x0102].value == 0

    def test_nonbinary_raises(self):
        """An array containing values other than 0 and 1 is rejected."""
        ds = pydicom.Dataset()
        with pytest.raises(ValueError, match="binary"):
            set_overlay_array(ds, 0x6000, np.full((4, 4), 2, dtype="u1"))

    def test_bad_shape_raises(self):
        """A 1D or 4D array is rejected."""
        ds = pydicom.Dataset()
        with pytest.raises(ValueError, match="rows, columns"):
            set_overlay_array(ds, 0x6000, np.zeros(16, dtype="u1"))

        with pytest.raises(ValueError, match="rows, columns"):
            set_overlay_array(ds, 0x6000, np.zeros((2, 2, 2, 2), dtype="u1"))

    def test_bad_group_raises(self):
        """An overlay group outside 0x6000-0x60FF is rejected."""
        ds = pydicom.Dataset()
        with pytest.raises(ValueError, match="0x6000 and 0x60FF"):
            set_overlay_array(ds, 0x6100, np.zeros((2, 2), dtype="u1"))

    def test_bad_origin_raises(self):
        """An invalid overlay origin or frame origin is rejected."""
        ds = pydicom.Dataset()
        mask = np.zeros((2, 2), dtype="u1")
        with pytest.raises(ValueError, match="origin"):
            set_overlay_array(ds, 0x6000, mask, origin=(1, 2, 3))

        with pytest.raises(ValueError, match="frame number"):
            set_overlay_array(ds, 0x6000, mask, image_frame_origin=0)

    def test_dataset_method(self):
        """Dataset.set_overlay_array delegates to the handler."""
        ds = pydicom.Dataset()
        mask = np.eye(8, dtype="u1")
        ds.set_overlay_array(0x6000, mask)
        assert np.array_equal(ds.overlay_array(0x6000), mask)

        with pytest.raises(ValueError, match="0x6000 and 0x60FF"):
            ds.set_overlay_array(0x6100, mask)


@pytest.mark.skipif(not HAVE_NP, reason="Numpy is not available")
class TestNumpy_SetEmbeddedOverlay:
    """Tests for set_overlay_array with an embedded pixel bit plane."""

    @staticmethod
    def _pixel_ds(bits_stored=12, nr_frames=1, pixels=None, tsyntax=EXPL):
        """Return a 4x4 per frame 16-bit uncompressed dataset."""
        size = 4 * 4 * nr_frames
        if pixels is None:
            rng = np.random.default_rng(1)
            pixels = rng.integers(0, 2**bits_stored, size=size, dtype=np.uint16)

        ds = pydicom.Dataset()
        ds.preamble = b"\x00" * 128
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = tsyntax
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        ds.file_meta.MediaStorageSOPInstanceUID = (
            "1.2.826.0.1.3680043.8.498.1"
        )
        ds.Rows = 4
        ds.Columns = 4
        if nr_frames > 1:
            ds.NumberOfFrames = nr_frames
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = bits_stored
        ds.HighBit = bits_stored - 1
        ds.PixelRepresentation = 0
        ds.PixelData = pixels.tobytes()
        return ds, pixels

    def test_embedded_roundtrip(self):
        """The bit plane and all non-target bits shall be correct."""
        ds, pixels = self._pixel_ds()
        mask = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
            dtype="u1",
        )
        set_overlay_array(ds, 0x6000, mask, bit_plane=15)

        result = np.frombuffer(ds.PixelData, dtype="<u2")
        assert np.array_equal(result & 0x7FFF, pixels)
        assert np.array_equal((result >> 15) & 1, mask.ravel())
        assert (0x6000, 0x3000) not in ds
        assert ds[0x6000, 0x0100].value == 16
        assert ds[0x6000, 0x0102].value == 15
        assert ds[0x6000, 0x0010].value == 4
        assert ds[0x6000, 0x0011].value == 4
        assert list(ds[0x6000, 0x0050].value) == [1, 1]

    def test_embedded_multiframe_mapping(self):
        """Overlay frames shall map using the image frame origin."""
        ds, pixels = self._pixel_ds(nr_frames=3)
        mask = np.ones((2, 4, 4), dtype="u1")
        set_overlay_array(
            ds, 0x6000, mask, image_frame_origin=2, bit_plane=15
        )

        result = np.frombuffer(ds.PixelData, dtype="<u2").reshape(3, 16)
        assert np.array_equal(result[0], pixels[:16])
        assert np.array_equal(result[1] & 0x7FFF, pixels[16:32])
        assert np.array_equal(result[2] & 0x7FFF, pixels[32:48])
        assert np.array_equal(result[1:] >> 15, np.ones((2, 16), dtype="u1"))
        assert ds[0x6000, 0x0015].value == 2
        assert ds[0x6000, 0x0051].value == 2

    def test_embedded_origin_offset(self):
        """A 2x2 mask at origin (3,3) maps to the bottom-right pixels."""
        ds, pixels = self._pixel_ds()
        mask = np.ones((2, 2), dtype="u1")
        set_overlay_array(ds, 0x6000, mask, origin=(3, 3), bit_plane=15)

        result = np.frombuffer(ds.PixelData, dtype="<u2").reshape(4, 4)
        expected = np.zeros((4, 4), dtype="u1")
        expected[2:, 2:] = 1
        assert np.array_equal((result >> 15) & 1, expected)
        assert np.array_equal(result & 0x7FFF, pixels.reshape(4, 4))
        assert list(ds[0x6000, 0x0050].value) == [3, 3]

    def test_embedded_overwrite_valid_bits_raises(self):
        """A bit position below Bits Stored shall be refused."""
        ds, _ = self._pixel_ds(bits_stored=12)
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(ValueError, match="valid pixel bit"):
            set_overlay_array(ds, 0x6000, mask, bit_plane=11)

    def test_embedded_bit_position_at_bits_stored_allowed(self):
        """A bit position equal to Bits Stored is unused by the pixels."""
        ds, pixels = self._pixel_ds(bits_stored=12)
        mask = np.ones((4, 4), dtype="u1")
        set_overlay_array(ds, 0x6000, mask, bit_plane=12)
        result = np.frombuffer(ds.PixelData, dtype="<u2")
        assert np.array_equal(result & 0x0FFF, pixels)
        assert np.array_equal((result >> 12) & 1, mask.ravel())

    def test_embedded_bad_bit_position_raises(self):
        """A bit position outside the allocated bits shall be refused."""
        ds, _ = self._pixel_ds()
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(ValueError, match="bit position"):
            set_overlay_array(ds, 0x6000, mask, bit_plane=16)

    def test_embedded_compressed_raises(self):
        """Compressed pixel data shall be clearly rejected."""
        ds, _ = self._pixel_ds(tsyntax=JPEG2000)
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(NotImplementedError, match="compressed"):
            set_overlay_array(ds, 0x6000, mask, bit_plane=15)

    def test_embedded_big_endian_raises(self):
        """Big endian pixel data shall be rejected."""
        ds, _ = self._pixel_ds(tsyntax=ExplicitVRBigEndian)
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(NotImplementedError, match="big endian"):
            set_overlay_array(ds, 0x6000, mask, bit_plane=15)

    def test_embedded_missing_element_raises(self):
        """Missing Image Pixel module elements shall be reported."""
        ds, _ = self._pixel_ds()
        del ds.BitsStored
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(AttributeError, match="BitsStored"):
            set_overlay_array(ds, 0x6000, mask, bit_plane=15)

    def test_embedded_multisample_raises(self):
        """A dataset with multiple samples per pixel shall be rejected."""
        ds, _ = self._pixel_ds()
        ds.SamplesPerPixel = 3
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(ValueError, match="sample per pixel"):
            set_overlay_array(ds, 0x6000, mask, bit_plane=15)

    def test_embedded_bad_pixel_length_raises(self):
        """Pixel data that doesn't match the dimensions shall be rejected."""
        ds, _ = self._pixel_ds()
        ds.PixelData = ds.PixelData[:-2]
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(ValueError, match="pixel data length"):
            set_overlay_array(ds, 0x6000, mask, bit_plane=15)

    def test_embedded_8bit_pixels(self):
        """An unused high bit of 8-bit pixels may carry the overlay."""
        rng = np.random.default_rng(2)
        pixels = rng.integers(0, 16, size=16, dtype=np.uint8)
        ds = pydicom.Dataset()
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = EXPL
        ds.Rows = 4
        ds.Columns = 4
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 8
        ds.BitsStored = 4
        ds.HighBit = 3
        ds.PixelRepresentation = 0
        ds.PixelData = pixels.tobytes()
        mask = np.eye(4, dtype="u1")
        set_overlay_array(ds, 0x6000, mask, bit_plane=7)

        result = np.frombuffer(ds.PixelData, dtype="u1")
        assert np.array_equal(result & 0x7F, pixels)
        assert np.array_equal((result >> 7) & 1, mask.ravel())
        assert ds[0x6000, 0x0100].value == 8
        assert ds[0x6000, 0x0102].value == 7

    def test_embedded_negative_origin_crop(self):
        """A mask starting above/left of the pixels is cropped, not shifted."""
        ds, pixels = self._pixel_ds()
        mask = np.ones((3, 3), dtype="u1")
        # Origin (0, 0): first overlay row/column lies outside the image
        set_overlay_array(
            ds, 0x6000, mask, origin=(0, 0), bit_plane=15, crop=True
        )

        result = np.frombuffer(ds.PixelData, dtype="<u2").reshape(4, 4)
        expected = np.zeros((4, 4), dtype="u1")
        expected[:2, :2] = 1
        assert np.array_equal((result >> 15) & 1, expected)
        assert np.array_equal(result & 0x7FFF, pixels.reshape(4, 4))
        assert ds[0x6000, 0x0010].value == 2
        assert ds[0x6000, 0x0011].value == 2
        assert list(ds[0x6000, 0x0050].value) == [1, 1]

    def test_embedded_out_of_bounds_raises(self):
        """A mask extending past the pixels is rejected unless cropping."""
        ds, pixels = self._pixel_ds()
        mask = np.ones((3, 4), dtype="u1")
        with pytest.raises(ValueError, match="extends beyond"):
            set_overlay_array(
                ds, 0x6000, mask, origin=(3, 2), bit_plane=15
            )

        # Nothing was written when the request was rejected
        assert np.frombuffer(ds.PixelData, dtype="<u2").tolist() == pixels.tolist()

    def test_embedded_spatial_crop(self):
        """Only the overlapping region is written when cropping."""
        ds, pixels = self._pixel_ds()
        mask = np.ones((3, 4), dtype="u1")
        set_overlay_array(
            ds, 0x6000, mask, origin=(3, 2), bit_plane=15, crop=True
        )

        result = np.frombuffer(ds.PixelData, dtype="<u2").reshape(4, 4)
        expected = np.zeros((4, 4), dtype="u1")
        expected[2:, 1:] = 1
        assert np.array_equal((result >> 15) & 1, expected)
        assert np.array_equal(result & 0x7FFF, pixels.reshape(4, 4))
        # The synced overlay describes the cropped mask, not a shifted one
        assert ds[0x6000, 0x0010].value == 2
        assert ds[0x6000, 0x0011].value == 3
        assert list(ds[0x6000, 0x0050].value) == [3, 2]

    def test_embedded_no_overlap_raises(self):
        """A mask entirely outside the pixels is rejected even cropping."""
        ds, _ = self._pixel_ds()
        mask = np.ones((2, 2), dtype="u1")
        with pytest.raises(ValueError, match="does not overlap"):
            set_overlay_array(
                ds, 0x6000, mask, origin=(5, 1), bit_plane=15, crop=True
            )

    def test_embedded_frame_origin_out_of_range_raises(self):
        """An image frame origin beyond the last frame shall be rejected."""
        ds, _ = self._pixel_ds(nr_frames=3)
        mask = np.ones((1, 2, 2), dtype="u1")
        with pytest.raises(ValueError, match="Image Frame Origin"):
            set_overlay_array(
                ds, 0x6000, mask, image_frame_origin=4, bit_plane=15
            )

    def test_embedded_frame_overflow_raises(self):
        """Overlay frames past the last image frame are rejected."""
        ds, _ = self._pixel_ds(nr_frames=3)
        mask = np.ones((3, 2, 2), dtype="u1")
        with pytest.raises(ValueError, match="image frames"):
            set_overlay_array(
                ds, 0x6000, mask, image_frame_origin=2, bit_plane=15
            )

    def test_embedded_frame_crop(self):
        """Only the overlapping frames are written when cropping."""
        ds, pixels = self._pixel_ds(nr_frames=3)
        mask = np.ones((3, 4, 4), dtype="u1")
        set_overlay_array(
            ds,
            0x6000,
            mask,
            image_frame_origin=2,
            bit_plane=15,
            crop=True,
        )

        result = np.frombuffer(ds.PixelData, dtype="<u2").reshape(3, 16)
        assert np.array_equal(result[0], pixels[:16])
        assert np.array_equal(result[1:] >> 15, np.ones((2, 16), dtype="u1"))
        assert ds[0x6000, 0x0015].value == 2
        assert ds[0x6000, 0x0051].value == 2

    def test_embedded_overwrites_existing_plane(self):
        """Writing again replaces the plane without touching other bits."""
        ds, pixels = self._pixel_ds()
        first = np.ones((4, 4), dtype="u1")
        set_overlay_array(ds, 0x6000, first, bit_plane=15)
        second = np.zeros((4, 4), dtype="u1")
        set_overlay_array(ds, 0x6000, second, bit_plane=15)

        result = np.frombuffer(ds.PixelData, dtype="<u2")
        assert np.array_equal(result, pixels)
        assert (0x6000, 0x3000) not in ds

    def test_embedded_file_roundtrip(self):
        """The modified pixel data and elements survive writing to file."""
        import io

        from pydicom.filewriter import dcmwrite
        from pydicom.filereader import dcmread

        ds, _ = self._pixel_ds()
        mask = np.eye(4, dtype="u1")
        set_overlay_array(ds, 0x6000, mask, bit_plane=15)

        buffer = io.BytesIO()
        dcmwrite(buffer, ds, enforce_file_format=True)
        buffer.seek(0)
        read = dcmread(buffer)
        plane = (np.frombuffer(read.PixelData, dtype="<u2") >> 15) & 1
        assert np.array_equal(plane.reshape(4, 4), mask)
        assert read[0x6000, 0x0100].value == 16
        assert read[0x6000, 0x0102].value == 15
