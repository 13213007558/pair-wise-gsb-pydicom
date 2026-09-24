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

from typing import TYPE_CHECKING, cast, Any

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

from pydicom.misc import warn_and_log
from pydicom.pixels.utils import pack_bits, unpack_bits

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


def extract_overlay(ds: "Dataset", group: int) -> "Dataset":
    """Extract an overlay plane stored in the unused bits of the *Pixel Data*
    into a standalone (``group``,3000) *Overlay Data* element.

    .. versionadded:: 3.1

    The overlay plane identified by `group` is read from bit
    (``group``,0102) *Overlay Bit Position* of each pixel cell in
    (7FE0,0010) *Pixel Data*, packed into a new (``group``,3000) *Overlay
    Data* element, and the corresponding bits in the *Pixel Data* are
    cleared. Only the bits belonging to the extracted overlay are modified;
    the stored pixel values, their signed representation and any bits used
    by other overlay planes are left unchanged.

    **Supported data**

    +------------------------------------------------+--------------+
    | Element                                        | Supported    |
    +-------------+---------------------------+------+ values       |
    | Tag         | Keyword                   | Type |              |
    +=============+===========================+======+==============+
    | (0028,0100) | BitsAllocated             | 1    | 8, 16, 32,   |
    |             |                           |      | 64           |
    +-------------+---------------------------+------+--------------+
    | (0028,0004) | SamplesPerPixel           | 1    | 1            |
    +-------------+---------------------------+------+--------------+
    | (60xx,0100) | OverlayBitsAllocated      | 1    | same as      |
    |             |                           |      | BitsAllocated  |
    +-------------+---------------------------+------+--------------+
    | (60xx,0102) | OverlayBitPosition        | 1    | N >=         |
    |             |                           |      | BitsStored   |
    +-------------+---------------------------+------+--------------+

    Only datasets using an uncompressed (native) transfer syntax are
    supported.

    Parameters
    ----------
    ds : Dataset
        The :class:`pydicom.dataset.Dataset` containing an Overlay Plane
        module with the overlay data embedded in the *Pixel Data*.
    group : int
        The group part of the overlay element tags, e.g. ``0x6000``,
        ``0x6002``, etc. Must be an even number between ``0x6000`` and
        ``0x60FE``.

    Returns
    -------
    Dataset
        The same `ds`, modified in place:

        * (``group``,3000) *Overlay Data* is added, with the overlay bits
          packed continuously across bytes and frames (first pixel in the
          least significant bit, see :dcm:`PS3.5 Section 8.1.1
          <part05/chapter_8.html#sect_8.1.1>`)
        * (``group``,0100) *Overlay Bits Allocated* is set to ``1`` and
          (``group``,0102) *Overlay Bit Position* to ``0``
        * The overlay's bits in (7FE0,0010) *Pixel Data* are set to ``0``

    Raises
    ------
    AttributeError
        If `ds` is missing a required element or the transfer syntax cannot
        be determined.
    ValueError
        If the dataset uses a compressed transfer syntax, the overlay is
        already stored as *Overlay Data*, the overlay bit position falls
        within the stored pixel bits or outside the allocated bits, the
        overlay metadata is contradictory, or the overlay region lies
        outside the stored pixel data.

    References
    ----------

    * DICOM Standard, Part 3, Sections :dcm:`C.9.2<part03/sect_C.9.2.html>`
      and :dcm:`C.9.3<part03/sect_C.9.3.html>`
    * DICOM Standard, Part 5, :dcm:`Section 8.1.1
      <part05/chapter_8.html#sect_8.1.1>`
    """
    if not HAVE_NP:
        raise ImportError("The overlay data handler requires numpy")

    if not 0x6000 <= group <= 0x60FE or group % 2:
        raise ValueError(
            "The overlay group must be an even number between 0x6000 and "
            f"0x60FE, got 0x{group:04X}"
        )

    # Check required overlay elements
    elem = {
        "OverlayRows": ds.get((group, 0x0010), None),
        "OverlayColumns": ds.get((group, 0x0011), None),
        "OverlayBitsAllocated": ds.get((group, 0x0100), None),
        "OverlayBitPosition": ds.get((group, 0x0102), None),
    }
    missing = [kk for kk, vv in elem.items() if vv is None]
    if missing:
        raise AttributeError(
            "Unable to extract the overlay as the following required "
            f"elements are missing from the dataset: {', '.join(missing)}"
        )

    # Check required image pixel module elements
    required = [
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "SamplesPerPixel",
        "PixelRepresentation",
        "PixelData",
    ]
    missing = [kw for kw in required if kw not in ds]
    if missing:
        raise AttributeError(
            "Unable to extract the overlay as the following required "
            f"elements are missing from the dataset: {', '.join(missing)}"
        )

    # Only uncompressed (native) transfer syntaxes are supported
    file_meta = getattr(ds, "file_meta", None)
    tsyntax = getattr(file_meta, "TransferSyntaxUID", None)
    if tsyntax is not None:
        if tsyntax.is_compressed:
            raise ValueError(
                "Unable to extract the overlay from the pixel data of a "
                "dataset using the compressed transfer syntax "
                f"'{tsyntax.name}'"
            )
        is_little_endian = tsyntax.is_little_endian
    else:
        is_little_endian = ds.original_encoding[1]

    if is_little_endian is None:
        raise AttributeError(
            "Unable to determine the endianness of the dataset, please set "
            "an appropriate Transfer Syntax UID in "
            f"'{type(ds).__name__}.file_meta'"
        )

    if (group, 0x3000) in ds:
        raise ValueError(
            f"Unable to extract the overlay as (0x{group:04X},0x3000) "
            "'Overlay Data' is already present in the dataset"
        )

    ov = {kk: vv.value for kk, vv in elem.items()}
    bits_allocated = int(ds.BitsAllocated)
    bits_stored = int(ds.BitsStored)

    if int(ov["OverlayBitsAllocated"]) == 1:
        raise ValueError(
            "Unable to extract the overlay as a value of 1 for "
            f"(0x{group:04X},0100) 'Overlay Bits Allocated' indicates the "
            "overlay is not stored in the pixel data"
        )

    if int(ov["OverlayBitsAllocated"]) != bits_allocated:
        raise ValueError(
            f"Unable to extract the overlay as (0x{group:04X},0100) "
            f"'Overlay Bits Allocated' ({ov['OverlayBitsAllocated']}) "
            f"doesn't match (0028,0100) 'Bits Allocated' ({bits_allocated})"
        )

    if bits_allocated % 8:
        raise ValueError(
            "Unable to extract the overlay as only uncompressed integer "
            f"pixel data is supported, got a value of {bits_allocated} for "
            "(0028,0100) 'Bits Allocated'"
        )

    if int(ds.SamplesPerPixel) != 1:
        raise ValueError(
            "Unable to extract the overlay as only a value of 1 for "
            "(0028,0004) 'Samples per Pixel' is supported, got "
            f"{ds.SamplesPerPixel}"
        )

    bit_position = int(ov["OverlayBitPosition"])
    if not 0 <= bit_position < bits_allocated:
        raise ValueError(
            f"Unable to extract the overlay as (0x{group:04X},0102) "
            f"'Overlay Bit Position' ({bit_position}) is outside the "
            f"allocated bits (0 to {bits_allocated - 1})"
        )

    if bit_position < bits_stored:
        raise ValueError(
            f"Unable to extract the overlay as (0x{group:04X},0102) "
            f"'Overlay Bit Position' ({bit_position}) falls within the "
            f"stored pixel bits (0 to {bits_stored - 1})"
        )

    # Optional overlay elements
    origin = ds.get((group, 0x0050), None)
    origin = [1, 1] if origin is None else [int(v) for v in origin.value]
    if len(origin) != 2:
        raise ValueError(
            f"Unable to extract the overlay as (0x{group:04X},0050) "
            f"'Overlay Origin' must contain exactly 2 values, got "
            f"{len(origin)}"
        )

    nr_overlay_frames = ds.get((group, 0x0015), None)
    nr_overlay_frames = 1 if nr_overlay_frames is None else int(nr_overlay_frames.value)
    frame_origin = ds.get((group, 0x0051), None)
    frame_origin = 1 if frame_origin is None else int(frame_origin.value)

    overlay_rows = int(ov["OverlayRows"])
    overlay_columns = int(ov["OverlayColumns"])
    if overlay_rows < 1 or overlay_columns < 1:
        raise ValueError(
            "Unable to extract the overlay as the values for "
            f"(0x{group:04X},0010) 'Overlay Rows' and (0x{group:04X},0011) "
            f"'Overlay Columns' must be greater than 0, got "
            f"{overlay_rows} and {overlay_columns}"
        )
    if nr_overlay_frames < 1:
        raise ValueError(
            "Unable to extract the overlay as the value for "
            f"(0x{group:04X},0015) 'Number of Frames in Overlay' must be "
            f"greater than 0, got {nr_overlay_frames}"
        )
    if frame_origin < 1:
        raise ValueError(
            "Unable to extract the overlay as the value for "
            f"(0x{group:04X},0051) 'Image Frame Origin' must be greater "
            f"than 0, got {frame_origin}"
        )

    # Map the overlay frames onto the image frames
    nr_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    first_frame = frame_origin - 1
    if first_frame + nr_overlay_frames > nr_frames:
        raise ValueError(
            f"Unable to extract the overlay as {nr_overlay_frames} overlay "
            f"frame(s) starting at image frame {frame_origin} exceed the "
            f"number of image frames ({nr_frames})"
        )

    # Determine where the overlay region intersects the image
    rows = int(ds.Rows)
    columns = int(ds.Columns)
    row_0 = origin[1] - 1  # 0-based row of the overlay's top-left pixel
    col_0 = origin[0] - 1  # 0-based column of the overlay's top-left pixel
    row_start, row_end = max(row_0, 0), min(row_0 + overlay_rows, rows)
    col_start, col_end = max(col_0, 0), min(col_0 + overlay_columns, columns)
    if row_start >= row_end or col_start >= col_end:
        raise ValueError(
            "Unable to extract the overlay as the overlay region "
            f"(origin {origin}, {overlay_rows} row(s) x {overlay_columns} "
            f"column(s)) lies outside the {rows} x {columns} pixel data"
        )

    # View the pixel data as unsigned integers
    bytes_per_pixel = bits_allocated // 8
    dtype = np.dtype(f"uint{bits_allocated}")
    dtype = dtype.newbyteorder("<" if is_little_endian else ">")
    expected_len = nr_frames * rows * columns * bytes_per_pixel
    pixel_data = cast(bytes, ds.PixelData)
    if len(pixel_data) < expected_len:
        raise ValueError(
            "The length of the pixel data in the dataset "
            f"({len(pixel_data)} bytes) is less than the expected length "
            f"({expected_len} bytes). The dataset may be corrupted."
        )

    arr = np.frombuffer(pixel_data[:expected_len], dtype=dtype).reshape(
        nr_frames, rows, columns
    )
    arr = arr.copy()

    # Extract the overlay bits from the region they apply to
    overlay = np.zeros((nr_overlay_frames, overlay_rows, overlay_columns), dtype="u1")
    region = arr[
        first_frame : first_frame + nr_overlay_frames,
        row_start:row_end,
        col_start:col_end,
    ]
    overlay[
        :,
        row_start - row_0 : row_end - row_0,
        col_start - col_0 : col_end - col_0,
    ] = ((region >> bit_position) & 1).astype("u1")

    # Clear only the extracted overlay's bits from the pixel data
    mask = dtype.type(1 << bit_position)
    region &= ~mask
    ds.PixelData = arr.tobytes() + pixel_data[expected_len:]

    # Add the standalone overlay data, packed continuously across frames
    ds.add_new((group, 0x3000), "OW", pack_bits(overlay.ravel()))
    ds[group, 0x0100].value = 1  # Overlay Bits Allocated
    ds[group, 0x0102].value = 0  # Overlay Bit Position

    return ds
