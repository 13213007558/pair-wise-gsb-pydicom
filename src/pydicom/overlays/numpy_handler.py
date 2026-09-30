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
from pydicom.uid import UID

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


def _validate_overlay_array(arr: "np.ndarray") -> "np.ndarray":
    """Return `arr` as a (frames, rows, columns) ``uint8`` array of 0s and 1s."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    if arr.ndim != 3 or 0 in arr.shape:
        raise ValueError(
            "The overlay array must be shaped (rows, columns) or "
            f"(frames, rows, columns) and be non-empty, got shape {arr.shape}"
        )

    if not np.array_equal(arr, arr.astype(bool)):
        raise ValueError("The overlay array must only contain binary values (0 or 1)")

    return arr.astype("u1")


def _sync_overlay_elements(
    ds: "Dataset",
    group: int,
    arr: "np.ndarray",
    bits_allocated: int,
    bit_position: int,
    image_frames: int,
) -> None:
    """Synchronise the overlay plane module elements in `ds` with `arr`."""
    nr_frames, nr_rows, nr_columns = arr.shape

    ds.add_new((group, 0x0010), "US", nr_rows)  # OverlayRows
    ds.add_new((group, 0x0011), "US", nr_columns)  # OverlayColumns
    ds.add_new((group, 0x0015), "IS", nr_frames)  # NumberOfFramesInOverlay
    ds.add_new((group, 0x0100), "US", bits_allocated)  # OverlayBitsAllocated
    ds.add_new((group, 0x0102), "US", bit_position)  # OverlayBitPosition
    if (group, 0x0040) not in ds:
        ds.add_new((group, 0x0040), "CS", "R")  # OverlayType
    if (group, 0x0050) not in ds:
        ds.add_new((group, 0x0050), "SS", [1, 1])  # OverlayOrigin
    if (nr_frames > 1 or image_frames > 1) and (group, 0x0051) not in ds:
        ds.add_new((group, 0x0051), "US", 1)  # ImageFrameOrigin


def _set_overlay_data(ds: "Dataset", group: int, arr: "np.ndarray") -> None:
    """Write `arr` to the dataset as standalone (60xx,3000) *Overlay Data*."""
    # pack_bits() packs the bit stream continuously (row boundaries are not
    #   byte aligned) and appends a trailing NULL padding byte for odd length
    #   data, as required by PS3.5 Section 8.1.1
    ds.add_new((group, 0x3000), "OW", pack_bits(arr, pad=True))
    _sync_overlay_elements(
        ds, group, arr, bits_allocated=1, bit_position=0, image_frames=1
    )

    # Re-read the written data to confirm the round-trip gives the input mask
    read_back = get_overlay_array(ds, group)
    if not np.array_equal(read_back.reshape(arr.shape), arr):
        raise RuntimeError(
            "Verification failed: the overlay data read back from the dataset "
            "doesn't match the array that was written"
        )


def _set_embedded_overlay(
    ds: "Dataset",
    group: int,
    arr: "np.ndarray",
    bit_position: "int | None",
    out_of_bounds: str,
) -> None:
    """Write `arr` to a bit plane of the (7FE0,0010) *Pixel Data*."""
    if "PixelData" not in ds:
        raise ValueError(
            "Unable to write an embedded overlay as the dataset has no "
            "(7FE0,0010) 'Pixel Data' element"
        )

    elem = ds["PixelData"]
    file_meta = getattr(ds, "file_meta", {})
    tsyntax = UID(file_meta.get("TransferSyntaxUID", "")) if file_meta else None
    if getattr(elem, "is_undefined_length", False) or (
        tsyntax is not None and tsyntax.is_compressed
    ):
        raise ValueError(
            "Unable to write an embedded overlay as the pixel data is "
            "encapsulated (compressed); decompress the dataset first"
        )

    missing = [
        kw for kw in ("Rows", "Columns", "BitsAllocated", "BitsStored") if kw not in ds
    ]
    if missing:
        raise AttributeError(
            "Unable to write an embedded overlay as the following required "
            f"elements are missing from the dataset: {', '.join(missing)}"
        )

    if getattr(ds, "SamplesPerPixel", 1) != 1:
        raise ValueError(
            "Unable to write an embedded overlay as only images with "
            "(0028,0002) 'Samples per Pixel' of 1 are supported"
        )

    bits_allocated = int(ds.BitsAllocated)
    if bits_allocated not in (8, 16):
        raise ValueError(
            "Unable to write an embedded overlay as only (0028,0100) 'Bits "
            f"Allocated' of 8 or 16 is supported, got {bits_allocated}"
        )

    if bit_position is None:
        elem_bp = ds.get((group, 0x0102), None)
        if elem_bp is None:
            raise ValueError(
                "Unable to write an embedded overlay as no 'bit_position' was "
                "given and the dataset has no (60xx,0102) 'Overlay Bit "
                "Position' element"
            )
        bit_position = int(elem_bp.value)

    if not 0 <= bit_position < bits_allocated:
        raise ValueError(
            f"The overlay bit position {bit_position} is invalid for "
            f"(0028,0100) 'Bits Allocated' of {bits_allocated}"
        )

    # Only allow the write if the target bit plane provably doesn't overlap
    #   any of the used pixel bits
    if bit_position < int(ds.BitsStored):
        raise ValueError(
            "Unable to write an embedded overlay at bit position "
            f"{bit_position} as (0028,0102) 'Bits Stored' is "
            f"{int(ds.BitsStored)}; writing would overwrite valid pixel bits"
        )

    # Map the overlay plane onto the image using (60xx,0050) 'Overlay Origin'
    #   and (60xx,0051) 'Image Frame Origin' (both 1-based)
    elem_origin = ds.get((group, 0x0050), None)
    if elem_origin is None:
        origin_row = origin_col = 1
    else:
        origin_row, origin_col = (int(v) for v in elem_origin.value)

    elem_frame = ds.get((group, 0x0051), None)
    frame_origin = 1 if elem_frame is None else int(elem_frame.value)

    img_rows, img_columns = int(ds.Rows), int(ds.Columns)
    img_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    nr_frames, nr_rows, nr_columns = arr.shape

    # Image space (0-based) extent of the overlay plane
    f0, f1 = frame_origin - 1, frame_origin - 1 + nr_frames
    r0, r1 = origin_row - 1, origin_row - 1 + nr_rows
    c0, c1 = origin_col - 1, origin_col - 1 + nr_columns

    out = (
        f0 < 0
        or f1 > img_frames
        or r0 < 0
        or r1 > img_rows
        or c0 < 0
        or c1 > img_columns
    )
    if out and out_of_bounds == "raise":
        raise ValueError(
            "The overlay plane extends outside the image: with the current "
            "(60xx,0050) 'Overlay Origin' and (60xx,0051) 'Image Frame "
            f"Origin' it covers frames {f0 + 1}..{f1}, rows {r0 + 1}..{r1} "
            f"and columns {c0 + 1}..{c1} of a {img_frames} frame, "
            f"{img_rows} x {img_columns} image. Either correct the origin, "
            "reshape the overlay array or pass out_of_bounds='crop' to crop "
            "the overlay to the image"
        )

    if out:
        # Crop the overlay to the image bounds
        cf0, cf1 = max(f0, 0), min(f1, img_frames)
        cr0, cr1 = max(r0, 0), min(r1, img_rows)
        cc0, cc1 = max(c0, 0), min(c1, img_columns)
        if cf0 >= cf1 or cr0 >= cr1 or cc0 >= cc1:
            raise ValueError(
                "The overlay plane lies entirely outside the image; nothing "
                "was written"
            )
    else:
        cf0, cf1, cr0, cr1, cc0, cc1 = f0, f1, r0, r1, c0, c1

    little_endian = tsyntax is None or tsyntax.is_little_endian
    dtype = np.dtype("u1" if bits_allocated == 8 else "<u2" if little_endian else ">u2")

    raw = bytes(elem.value)
    nr_pixels = img_frames * img_rows * img_columns
    expected_len = nr_pixels * dtype.itemsize
    if len(raw) < expected_len:
        raise ValueError(
            f"The length of the pixel data ({len(raw)} bytes) is less than "
            f"the expected length ({expected_len} bytes) for a {img_frames} "
            f"frame, {img_rows} x {img_columns} image with {bits_allocated} "
            "bits allocated"
        )

    shape = (img_frames, img_rows, img_columns)
    pixels = np.frombuffer(raw[:expected_len], dtype=dtype).reshape(shape).copy()
    original = pixels.copy()

    plane = dtype.type(1) << bit_position
    for frame in range(cf0, cf1):
        mask = arr[
            frame - f0,
            cr0 - r0 : cr1 - r0,
            cc0 - c0 : cc1 - c0,
        ].astype(dtype)
        region = pixels[frame, cr0:cr1, cc0:cc1]
        pixels[frame, cr0:cr1, cc0:cc1] = (region & ~plane) | (mask << bit_position)

    # Check that no bit outside the target plane has changed
    if not np.array_equal(original & ~plane, pixels & ~plane):
        raise RuntimeError(  # pragma: no cover
            "Verification failed: writing the overlay would change pixel "
            "bits outside the target bit plane"
        )

    elem.value = pixels.tobytes() + raw[expected_len:]
    if hasattr(ds, "_pixel_array"):
        ds._pixel_array = None

    # Re-read the stored pixel data and verify the round-trip: all non-target
    #   bits must be identical and the target plane must hold the written mask
    stored = np.frombuffer(bytes(ds.PixelData)[:expected_len], dtype=dtype)
    stored = stored.reshape(shape)
    if not np.array_equal(original & ~plane, stored & ~plane):
        raise RuntimeError(
            "Verification failed: pixel bits outside the target bit plane "
            "changed when writing the overlay"
        )

    for frame in range(cf0, cf1):
        mask = arr[frame - f0, cr0 - r0 : cr1 - r0, cc0 - c0 : cc1 - c0]
        read_back = (stored[frame, cr0:cr1, cc0:cc1] >> bit_position) & 1
        if not np.array_equal(read_back, mask):
            raise RuntimeError(
                "Verification failed: the overlay bit plane read back from "
                "the pixel data doesn't match the array that was written"
            )

    _sync_overlay_elements(
        ds,
        group,
        arr,
        bits_allocated=bits_allocated,
        bit_position=bit_position,
        image_frames=img_frames,
    )
    # Embedded overlays must not have a separate (60xx,3000) 'Overlay Data'
    if (group, 0x3000) in ds:
        del ds[group, 0x3000]


def set_overlay_array(
    ds: "Dataset",
    group: int,
    arr: "np.ndarray",
    *,
    embedded: bool = False,
    bit_position: "int | None" = None,
    out_of_bounds: str = "raise",
) -> None:
    """Write a binary :class:`numpy.ndarray` `arr` to `ds` as an overlay plane.

    .. versionadded:: 3.1

    Parameters
    ----------
    ds : Dataset
        The :class:`Dataset` to write the overlay to.
    group : int
        The group part of the overlay element tags, e.g. ``0x6000``,
        ``0x6010``, etc. Must be between 0x6000 and 0x60FF.
    arr : numpy.ndarray
        The overlay as a binary array (containing only 0s and 1s) shaped
        ``(rows, columns)`` for a single frame overlay or
        ``(frames, rows, columns)`` for a multi-frame overlay.
    embedded : bool, optional
        If ``False`` (default) then write the overlay as standalone
        (60xx,3000) *Overlay Data* with (60xx,0100) *Overlay Bits Allocated*
        of 1 and (60xx,0102) *Overlay Bit Position* of 0. If ``True`` then
        instead write the overlay into the bit plane at `bit_position` of the
        (7FE0,0010) *Pixel Data* (the retired embedded overlay format). This
        modifies the image's pixel data and is therefore only performed when
        explicitly requested; it also requires the pixel data to be
        uncompressed and the target bit plane to lie at or above
        (0028,0102) *Bits Stored* so no valid pixel bits are overwritten.
    bit_position : int, optional
        The bit plane of the pixel data to write to when `embedded` is
        ``True``. If not given then the value of the dataset's existing
        (60xx,0102) *Overlay Bit Position* element is used.
    out_of_bounds : str, optional
        Only used when `embedded` is ``True``. Determines what happens when
        the overlay plane, as mapped into the image by (60xx,0050) *Overlay
        Origin* and (60xx,0051) *Image Frame Origin*, extends outside the
        image: ``'raise'`` (default) raises a :class:`ValueError`, ``'crop'``
        crops the overlay to the image bounds. The overlay is never silently
        shifted to fit.

    Raises
    ------
    ValueError
        If `arr` is not a binary array, if the overlay cannot be written
        safely, or if the overlay extends outside the image and
        `out_of_bounds` is ``'raise'``.
    RuntimeError
        If the post-write verification fails.
    """
    if not HAVE_NP:
        raise ImportError("The overlay data handler requires numpy")

    if group < 0x6000 or group > 0x60FF:
        raise ValueError(
            "The group part of the 'Overlay Data' element tag must be "
            "between 0x6000 and 0x60FF (inclusive)"
        )

    if out_of_bounds not in ("raise", "crop"):
        raise ValueError(
            f"Invalid 'out_of_bounds' value '{out_of_bounds}', must be "
            "'raise' or 'crop'"
        )

    arr = _validate_overlay_array(arr)

    if embedded:
        _set_embedded_overlay(ds, group, arr, bit_position, out_of_bounds)
    else:
        _set_overlay_data(ds, group, arr)
