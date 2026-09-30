# Copyright 2008-2019 pydicom authors. See LICENSE file for details.
"""Use the `numpy <https://numpy.org/>`_ package to convert supported *Overlay
Data* to a :class:`numpy.ndarray`.

**Supported data**

The numpy handler supports the conversion of data in the (60xx,3000)
*Overlay Data* element to a :class:`~numpy.ndarray` provided the
related :dcm:`Overlay Plane<part03/sect_C.9.2.html>` and :dcm:`Multi-frame
Overlay<part03/sect_C.9.3.html>` module elements have values given in the
table below.

+------------------------------------------------+--------------+
| Element                                        | Supported    |
+-------------+---------------------------+------+ values       |
| Tag         | Keyword                   | Type |              |
+=============+===========================+======+==============+
| (60xx,0010) | OverlayRows               | 1    | N > 0        |
+-------------+---------------------------+------+--------------+
| (60xx,0011) | OverlayColumns            | 1    | N > 0        |
+-------------+---------------------------+------+--------------+
| (60xx,0015) | NumberOfFramesInOverlay   | 1    | N > 0        |
+-------------+---------------------------+------+--------------+
| (60xx,0100) | OverlayBitsAllocated      | 1    | 1            |
+-------------+---------------------------+------+--------------+
| (60xx,0102) | OverlayBitPosition        | 1    | 0            |
+-------------+---------------------------+------+--------------+

"""

from typing import TYPE_CHECKING, Any, cast

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

from pydicom.dataelem import DataElement
from pydicom.misc import warn_and_log
from pydicom.pixels.utils import get_nr_frames, pack_bits, unpack_bits

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset
    from pydicom.dataelem import DataElement


HANDLER_NAME = "Numpy Overlay"
DEPENDENCIES = {"numpy": ("https://numpy.org/", "NumPy")}


def is_available() -> bool:
    """Return ``True`` if the handler has its dependencies met."""
    return HAVE_NP


def get_expected_length(elem: dict[str, Any], unit: str = "bytes") -> int:
    """Return the expected length (in terms of bytes or pixels) of the *Overlay
    Data*.

    +------------------------------------------------+-------------+
    | Element                                        | Required or |
    +-------------+---------------------------+------+ optional    |
    | Tag         | Keyword                   | Type |             |
    +=============+===========================+======+=============+
    | (60xx,0010) | OverlayRows               | 1    | Required    |
    +-------------+---------------------------+------+-------------+
    | (60xx,0011) | OverlayColumns            | 1    | Required    |
    +-------------+---------------------------+------+-------------+
    | (60xx,0015) | NumberOfFramesInOverlay   | 1    | Required    |
    +-------------+---------------------------+------+-------------+

    Parameters
    ----------
    elem : dict
        A :class:`dict` with the keys as the element keywords and values the
        corresponding element values (such as ``{'OverlayRows': 512, ...}``)
        for the elements listed in the table above.
    unit : str, optional
        If ``'bytes'`` then returns the expected length of the *Overlay Data*
        in whole bytes and NOT including an odd length trailing NULL padding
        byte. If ``'pixels'`` then returns the expected length of the *Overlay
        Data* in terms of the total number of pixels (default ``'bytes'``).

    Returns
    -------
    int
        The expected length of the *Overlay Data* in either whole bytes or
        pixels, excluding the NULL trailing padding byte for odd length data.
    """
    length: int = elem["OverlayRows"] * elem["OverlayColumns"]
    length *= elem["NumberOfFramesInOverlay"]

    if unit == "pixels":
        return length

    # Determine the nearest whole number of bytes needed to contain
    #   1-bit pixel data. e.g. 10 x 10 1-bit pixels is 100 bits, which
    #   are packed into 12.5 -> 13 bytes
    return length // 8 + (length % 8 > 0)


def reshape_overlay_array(elem: dict[str, Any], arr: "np.ndarray") -> "np.ndarray":
    """Return a reshaped :class:`numpy.ndarray` `arr`.

    +------------------------------------------------+--------------+
    | Element                                        | Supported    |
    +-------------+---------------------------+------+ values       |
    | Tag         | Keyword                   | Type |              |
    +=============+===========================+======+==============+
    | (60xx,0010) | OverlayRows               | 1    | N > 0        |
    +-------------+---------------------------+------+--------------+
    | (60xx,0011) | OverlayColumns            | 1    | N > 0        |
    +-------------+---------------------------+------+--------------+
    | (60xx,0015) | NumberOfFramesInOverlay   | 1    | N > 0        |
    +-------------+---------------------------+------+--------------+

    Parameters
    ----------
    elem : dict
        A :class:`dict` with the keys as the element keywords and values the
        corresponding element values (such as ``{'OverlayRows': 512, ...}``)
        for the elements listed in the table above.
    arr : numpy.ndarray
        A 1D array containing the overlay data.

    Returns
    -------
    numpy.ndarray
        A reshaped array containing the overlay data. The shape of the array
        depends on the contents of the dataset:

        * For single frame data (rows, columns)
        * For multi-frame data (frames, rows, columns)

    References
    ----------

    * DICOM Standard, Part 3, Sections :dcm:`C.9.2<part03/sect_C.9.2.html>`
      and :dcm:`C.9.3<part03/sect_C.9.3.html>`
    * DICOM Standard, Part 5, :dcm:`Section 8.2<part05/sect_8.2.html>`
    """
    if not HAVE_NP:
        raise ImportError("Numpy is required to reshape the overlay array.")

    nr_frames = elem["NumberOfFramesInOverlay"]
    nr_rows = elem["OverlayRows"]
    nr_columns = elem["OverlayColumns"]

    if nr_frames < 1:
        raise ValueError(
            f"Unable to reshape the overlay array as a value of {nr_frames} "
            "for (60xx,0015) 'Number of Frames in Overlay' is invalid."
        )

    if nr_frames > 1:
        return arr.reshape(nr_frames, nr_rows, nr_columns)

    return arr.reshape(nr_rows, nr_columns)


def get_overlay_array(ds: "Dataset", group: int) -> "np.ndarray":
    """Return a :class:`numpy.ndarray` of the *Overlay Data*.

    Parameters
    ----------
    ds : Dataset
        The :class:`Dataset` containing an Overlay Plane module and the
        *Overlay Data* to be converted.
    group : int
        The group part of the *Overlay Data* element tag, e.g. ``0x6000``,
        ``0x6010``, etc. Must be between 0x6000 and 0x60FF.

    Returns
    -------
    np.ndarray
        The contents of (`group`,3000) *Overlay Data* as an array.

    Raises
    ------
    AttributeError
        If `ds` is missing a required element.
    ValueError
        If the actual length of the overlay data doesn't match the expected
        length.
    """
    if not HAVE_NP:
        raise ImportError("The overlay data handler requires numpy")

    # Check required elements
    elem = {
        "OverlayData": ds.get((group, 0x3000), None),
        "OverlayBitsAllocated": ds.get((group, 0x0100), None),
        "OverlayRows": ds.get((group, 0x0010), None),
        "OverlayColumns": ds.get((group, 0x0011), None),
    }

    missing = [kk for kk, vv in elem.items() if vv is None]
    if missing:
        raise AttributeError(
            "Unable to convert the overlay data as the following required "
            f"elements are missing from the dataset: {', '.join(missing)}"
        )

    # Grab the element values
    elem_values = {kk: vv.value for kk, vv in elem.items()}

    # Add in if not present
    nr_frames: DataElement | None = ds.get((group, 0x0015), None)
    if nr_frames is None:
        elem_values["NumberOfFramesInOverlay"] = 1
    else:
        elem_values["NumberOfFramesInOverlay"] = nr_frames.value

    # Calculate the expected length of the pixel data (in bytes)
    #   Note: this does NOT include the trailing null byte for odd length data
    expected_len = get_expected_length(elem_values)

    # Check that the actual length of the pixel data is as expected
    actual_length = len(cast(bytes, elem_values["OverlayData"]))

    # Correct for the trailing NULL byte padding for odd length data
    padded_expected_len = expected_len + expected_len % 2
    if actual_length < padded_expected_len:
        if actual_length == expected_len:
            warn_and_log("The overlay data length is odd and misses a padding byte.")
        else:
            raise ValueError(
                "The length of the overlay data in the dataset "
                f"({actual_length} bytes) doesn't match the expected length "
                f"({padded_expected_len} bytes). The dataset may be corrupted "
                "or there may be an issue with the overlay data handler."
            )
    elif actual_length > padded_expected_len:
        # PS 3.5, Section 8.1.1
        warn_and_log(
            f"The length of the overlay data in the dataset ({actual_length} "
            "bytes) indicates it contains excess padding. "
            f"{actual_length - expected_len} bytes will be removed "
            "from the end of the data"
        )

    # Unpack the pixel data into a 1D ndarray, skipping any trailing padding
    nr_pixels = get_expected_length(elem_values, unit="pixels")
    arr = cast("np.ndarray", unpack_bits(elem_values["OverlayData"])[:nr_pixels])

    return reshape_overlay_array(elem_values, arr)


def _validate_overlay_arr(arr: "np.ndarray") -> "np.ndarray":
    """Return `arr` as a 3D binary numpy.ndarray.

    Parameters
    ----------
    arr : numpy.ndarray
        A 2D or 3D array containing only 0 and 1 values, shaped as
        '(rows, columns)' or '(frames, rows, columns)'.

    Returns
    -------
    numpy.ndarray
        `arr` reshaped to '(frames, rows, columns)' with dtype 'uint8'.

    Raises
    ------
    ValueError
        If `arr` isn't 2D or 3D or contains values other than 0 and 1.
    """
    arr = np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(
            "The overlay array must be shaped as (rows, columns) for a "
            "single frame or (frames, rows, columns) for multi-frame data"
        )

    if not np.array_equal(arr, arr.astype(bool)):
        raise ValueError(
            "Only binary arrays (containing ones or zeroes) can be written "
            "as overlay data"
        )

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    return arr.astype("u1")


def _set_overlay_elements(
    ds: "Dataset",
    group: int,
    rows: int,
    columns: int,
    nr_frames: int,
    origin: tuple[int, int],
    image_frame_origin: int,
    bits_allocated: int,
    bit_position: int,
) -> None:
    """Add or update the related Overlay Plane module elements."""
    ds.add_new((group, 0x0010), "US", rows)
    ds.add_new((group, 0x0011), "US", columns)
    ds.add_new((group, 0x0050), "SS", [int(origin[0]), int(origin[1])])
    ds.add_new((group, 0x0051), "US", int(image_frame_origin))
    ds.add_new((group, 0x0100), "US", int(bits_allocated))
    ds.add_new((group, 0x0102), "US", int(bit_position))

    if nr_frames > 1:
        ds.add_new((group, 0x0015), "IS", str(nr_frames))
    elif (group, 0x0015) in ds:
        del ds[group, 0x0015]


def _write_overlay_data(
    ds: "Dataset",
    group: int,
    arr: "np.ndarray",
    origin: tuple[int, int],
    image_frame_origin: int,
) -> None:
    """Bit-pack `arr` into the (group, 3000) Overlay Data element."""
    nr_frames, nr_rows, nr_columns = arr.shape

    # The bit stream is packed contiguously; pack_bits pads the final
    #   partial byte and adds a trailing NULL for odd length data
    packed = pack_bits(arr, pad=True)
    ds[group, 0x3000] = DataElement((group, 0x3000), "OW", packed)

    _set_overlay_elements(
        ds,
        group,
        nr_rows,
        nr_columns,
        nr_frames,
        origin,
        image_frame_origin,
        bits_allocated=1,
        bit_position=0,
    )

    # Verify the round-trip: re-reading the overlay must give the mask
    read_back = get_overlay_array(ds, group).reshape(arr.shape)
    if not np.array_equal(read_back, arr):
        raise RuntimeError(
            "Verification of the written overlay data failed: the value read "
            "back from the dataset does not match the overlay array"
        )


def _write_embedded_overlay(
    ds: "Dataset",
    group: int,
    arr: "np.ndarray",
    origin: tuple[int, int],
    image_frame_origin: int,
    bit_plane: int,
    crop: bool,
) -> None:
    """Write `arr` to bit plane `bit_plane` of the dataset's Pixel Data."""
    file_meta = getattr(ds, "file_meta", None)
    tsyntax = file_meta.get("TransferSyntaxUID", None) if file_meta else None
    if tsyntax is not None:
        if tsyntax.is_compressed:
            raise NotImplementedError(
                "Unable to embed an overlay in the pixel data as the dataset "
                f"uses the compressed transfer syntax '{tsyntax.name}'. "
                "Decompress the pixel data first or write a standalone "
                "Overlay Data element instead"
            )

        if not tsyntax.is_little_endian:
            raise NotImplementedError(
                "Unable to embed an overlay in big endian encoded pixel data"
            )

    required = {
        "PixelData": ds.get(0x7FE00010, None),
        "Rows": ds.get(0x00280010, None),
        "Columns": ds.get(0x00280011, None),
        "BitsAllocated": ds.get(0x00280100, None),
        "BitsStored": ds.get(0x00280101, None),
    }
    missing = [kk for kk, vv in required.items() if vv is None]
    if missing:
        raise AttributeError(
            "Unable to embed an overlay in the pixel data as the following "
            "required elements are missing from the dataset: "
            + ", ".join(missing)
        )

    samples_per_pixel = getattr(ds, "SamplesPerPixel", 1)
    if samples_per_pixel != 1:
        raise ValueError(
            "Unable to embed an overlay in the pixel data as only a single "
            "sample per pixel is supported"
        )

    bits_allocated = int(required["BitsAllocated"].value)
    bits_stored = int(required["BitsStored"].value)
    nr_image_rows = int(required["Rows"].value)
    nr_image_columns = int(required["Columns"].value)
    nr_image_frames = get_nr_frames(ds)

    if bits_allocated not in (8, 16):
        raise ValueError(
            "Unable to embed an overlay in the pixel data as a Bits "
            f"Allocated of {bits_allocated} is not supported"
        )

    if not 0 <= bit_plane < bits_allocated:
        raise ValueError(
            "The overlay bit position must be between 0 and "
            f"{bits_allocated - 1} for a Bits Allocated of {bits_allocated}, "
            f"got {bit_plane}"
        )

    if bit_plane < bits_stored:
        raise ValueError(
            f"Refusing to embed an overlay at bit position {bit_plane} as it "
            f"would overwrite the {bits_stored} valid pixel bit(s) "
            "(Bits Stored). Use a bit position greater than or equal to "
            "Bits Stored or write a standalone Overlay Data element"
        )

    raw = required["PixelData"].value
    if not isinstance(raw, (bytes, bytearray)):
        raise NotImplementedError(
            "Unable to embed an overlay as the pixel data is not available "
            "as uncompressed bytes"
        )

    bytes_per_pixel = bits_allocated // 8
    expected_length = (
        nr_image_rows * nr_image_columns * nr_image_frames * bytes_per_pixel
    )
    if len(raw) != expected_length:
        raise ValueError(
            f"The pixel data length ({len(raw)} bytes) doesn't match the "
            f"expected length ({expected_length} bytes), so the overlay "
            "cannot be safely mapped onto the pixels"
        )

    arr = arr.copy()
    nr_overlay_frames, nr_overlay_rows, nr_overlay_columns = arr.shape

    # Map the overlay frames onto the image frames (1-based origin).
    #   Out-of-range frames are rejected or cropped, never shifted.
    image_frame_origin = int(image_frame_origin)
    first_frame = image_frame_origin - 1
    if not 0 <= first_frame < nr_image_frames:
        raise ValueError(
            f"The Image Frame Origin {image_frame_origin} is outside the "
            f"range of the {nr_image_frames} image frame(s)"
        )

    frame_lo = max(0, -first_frame)
    frame_hi = min(nr_overlay_frames, nr_image_frames - first_frame)
    if frame_lo >= frame_hi:
        raise ValueError(
            "The overlay frames do not overlap with the image frames; there "
            "is no region to write"
        )

    if (frame_lo > 0 or frame_hi < nr_overlay_frames) and not crop:
        raise ValueError(
            "The overlay extends beyond the available image frames. Pass "
            "crop=True to write only the overlapping frames"
        )

    if frame_lo > 0 or frame_hi < nr_overlay_frames:
        arr = arr[frame_lo:frame_hi]
        image_frame_origin += frame_lo
        first_frame = image_frame_origin - 1

    # Map the overlay rows/columns onto the pixel matrix (1-based origin)
    first_row = int(origin[0]) - 1
    first_column = int(origin[1]) - 1

    row_lo = max(0, -first_row)
    row_hi = min(nr_overlay_rows, nr_image_rows - first_row)
    column_lo = max(0, -first_column)
    column_hi = min(nr_overlay_columns, nr_image_columns - first_column)

    if row_lo >= row_hi or column_lo >= column_hi:
        raise ValueError(
            "The overlay does not overlap with the pixel matrix at the given "
            "Overlay Origin; there is no region to write"
        )

    spatial_overflow = (
        row_lo > 0
        or row_hi < nr_overlay_rows
        or column_lo > 0
        or column_hi < nr_overlay_columns
    )
    if spatial_overflow and not crop:
        raise ValueError(
            "The overlay extends beyond the image pixel matrix. Pass "
            "crop=True to write only the overlapping region"
        )

    if spatial_overflow:
        arr = arr[:, row_lo:row_hi, column_lo:column_hi]
        origin = (origin[0] + row_lo, origin[1] + column_lo)

    nr_frames, nr_rows, nr_columns = arr.shape
    pixel_row0 = origin[0] - 1
    pixel_column0 = origin[1] - 1

    dtype = "<u2" if bits_allocated == 16 else "u1"
    pixels = np.frombuffer(bytes(raw), dtype=dtype).copy().reshape(
        nr_image_frames, nr_image_rows, nr_image_columns
    )
    before = pixels.copy()

    target_bit = (
        np.uint16(1 << bit_plane)
        if bits_allocated == 16
        else np.uint8(1 << bit_plane)
    )
    other_bits = np.array(
        np.iinfo(pixels.dtype).max ^ target_bit, dtype=pixels.dtype
    )

    for frame_index in range(nr_frames):
        region = pixels[
            first_frame + frame_index,
            pixel_row0 : pixel_row0 + nr_rows,
            pixel_column0 : pixel_column0 + nr_columns,
        ]
        region &= other_bits
        region |= arr[frame_index].astype(pixels.dtype) * target_bit

    # All non-target bits must be identical before and after the write
    if not np.array_equal(before & other_bits, pixels & other_bits):
        raise RuntimeError(
            "Verification of the embedded overlay failed: bits other than "
            f"bit position {bit_plane} were modified"
        )

    ds.PixelData = pixels.tobytes()

    # An embedded overlay is stored in the pixel data, not an Overlay Data
    if (group, 0x3000) in ds:
        del ds[group, 0x3000]

    _set_overlay_elements(
        ds,
        group,
        nr_rows,
        nr_columns,
        nr_frames,
        origin,
        image_frame_origin,
        bits_allocated=bits_allocated,
        bit_position=bit_plane,
    )

    # Verify the round-trip by extracting the bit plane again from the
    #   pixel data written back to the dataset
    read_back = np.empty_like(arr)
    reread = np.frombuffer(bytes(ds.PixelData), dtype=dtype).reshape(
        nr_image_frames, nr_image_rows, nr_image_columns
    )
    for frame_index in range(nr_frames):
        region = reread[
            first_frame + frame_index,
            pixel_row0 : pixel_row0 + nr_rows,
            pixel_column0 : pixel_column0 + nr_columns,
        ]
        read_back[frame_index] = (region & target_bit) >> bit_plane

    if not np.array_equal(read_back, arr):
        raise RuntimeError(
            "Verification of the embedded overlay failed: the bit plane read "
            "back from the pixel data does not match the overlay array"
        )


def set_overlay_array(
    ds: "Dataset",
    group: int,
    arr: "np.ndarray",
    *,
    origin: tuple[int, int] = (1, 1),
    image_frame_origin: int = 1,
    bit_plane: int | None = None,
    crop: bool = False,
) -> None:
    """Write an overlay mask to the dataset in `group`.

    .. versionadded:: 3.1

    By default the mask is bit-packed and written to the (group,3000)
    *Overlay Data* element and the related Overlay Plane module elements -
    *Overlay Rows*, *Overlay Columns*, *Number of Frames in Overlay*,
    *Overlay Origin*, *Image Frame Origin*, *Overlay Bits Allocated* and
    *Overlay Bit Position* - are added or updated to match the mask.

    Alternatively, if `bit_plane` is given then the mask is written to a bit
    plane of the dataset's existing *Pixel Data* using the retired embedded
    overlay convention. Because this modifies the pixel data it is only
    performed when explicitly requested and only when it can be proven that no
    valid pixel bits are overwritten:

    * The pixel data must be uncompressed and little endian encoded.
    * The dataset must have a single sample per pixel with *Bits Allocated*
      of 8 or 16.
    * `bit_plane` must be greater than or equal to *Bits Stored*.

    The mask is mapped onto the image using the (group,0050) *Overlay Origin*
    and (group,0051) *Image Frame Origin* values, both of which are 1-based.
    Parts of the mask that fall outside the pixel matrix are either rejected
    or, when `crop` is ``True``, cropped to the overlapping region. The mask
    is never shifted to make it fit.

    Parameters
    ----------
    ds : Dataset
        The :class:`Dataset` to write the overlay to.
    group : int
        The group part of the overlay element tag, e.g. ``0x6000``,
        ``0x6010``, etc. Must be between 0x6000 and 0x60FF.
    arr : numpy.ndarray
        A binary array (values of 0 and 1) shaped as ``(rows, columns)`` for
        a single frame or ``(frames, rows, columns)`` for a multi-frame
        overlay.
    origin : tuple[int, int], optional
        The ``(row, column)`` 1-based location of the first point of the
        overlay relative to the image pixels (default ``(1, 1)``). Only used
        to map the mask for an embedded overlay; stored in *Overlay Origin*.
    image_frame_origin : int, optional
        The 1-based number of the image frame the first overlay frame applies
        to (default ``1``). Only used to map the mask for an embedded
        overlay; stored in *Image Frame Origin*.
    bit_plane : int, optional
        If given, embed the overlay in the *Pixel Data* at this bit position
        instead of writing a standalone *Overlay Data* element.
    crop : bool, optional
        If ``True`` then cropped portions of the mask that fall outside the
        pixel matrix are dropped (embedded mode only). If ``False`` (default)
        then any out-of-bounds mask raises :class:`ValueError`.

    Raises
    ------
    ValueError
        If `group` is invalid, `arr` isn't a binary 2D/3D array, an embedded
        overlay would overwrite valid pixel bits, or the mask falls outside
        the pixel matrix with `crop` set to ``False``.
    AttributeError
        If the elements required for an embedded overlay are missing.
    NotImplementedError
        If an embedded overlay is requested for compressed, big endian or
        otherwise unsupported pixel data.
    RuntimeError
        If the verification of the written overlay fails.
    """
    if not HAVE_NP:
        raise ImportError("The overlay data handler requires numpy")

    if group < 0x6000 or group > 0x60FF:
        raise ValueError(
            "The group part of the overlay element tag must be between "
            "0x6000 and 0x60FF (inclusive)"
        )

    if image_frame_origin < 1:
        raise ValueError(
            "The image frame origin must be a 1-based frame number, got "
            f"{image_frame_origin}"
        )

    if len(origin) != 2:
        raise ValueError(
            "The overlay origin must be a (row, column) pair of integers"
        )

    mask = _validate_overlay_arr(arr)

    if bit_plane is None:
        _write_overlay_data(ds, group, mask, origin, image_frame_origin)
    else:
        _write_embedded_overlay(
            ds,
            group,
            mask,
            origin,
            image_frame_origin,
            int(bit_plane),
            crop,
        )
