"""Utilities for working with overlay planes stored in *Pixel Data*.

Older DICOM objects may store an overlay plane in unused bits of the
(7FE0,0010) *Pixel Data* element instead of in a (60xx,3000) *Overlay Data*
element. In that case (60xx,0102) *Overlay Bit Position* gives the bit of
each pixel cell (starting from 0) that contains the overlay data.
"""

from typing import TYPE_CHECKING, cast

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

from pydicom.dataelem import DataElement
from pydicom.misc import warn_and_log
from pydicom.pixels.utils import pack_bits, pixel_dtype
from pydicom.tag import Tag

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


def extract_overlay(ds: "Dataset", group: int) -> "np.ndarray":
    """Extract an overlay plane stored in the unused bits of (7FE0,0010)
    *Pixel Data* to a standalone (60xx,3000) *Overlay Data* element.

    .. versionadded:: 3.1

    The dataset is modified in place:

    * The overlay bits are read from the *Pixel Data* using (60xx,0102)
      *Overlay Bit Position*, taking (60xx,0010) *Overlay Rows*,
      (60xx,0011) *Overlay Columns*, (60xx,0050) *Overlay Origin*,
      (60xx,0051) *Overlay Frame Origin* and (60xx,0015) *Number of Frames
      in Overlay* into account. Parts of the overlay plane that fall outside
      the image (such as with a negative *Overlay Origin*) are set to 0.
    * The extracted data is bit-packed (first pixel in the least significant
      bit of the first byte) and written to (group,3000) *Overlay Data*.
    * (60xx,0100) *Overlay Bits Allocated* is set to 1 and (60xx,0102)
      *Overlay Bit Position* to 0.
    * Only the bits belonging to this overlay are cleared in the *Pixel
      Data*; the stored pixel values, their signedness and any other
      overlay bits are left unchanged.

    Only uncompressed, integer, single sample per pixel data is supported.

    Parameters
    ----------
    ds : Dataset
        The :class:`~pydicom.dataset.Dataset` containing the *Pixel Data*
        and the overlay plane module elements for the given group.
    group : int
        The group part of the overlay element tags, e.g. 0x6000, 0x6002,
        etc. Must be an even number between 0x6000 and 0x60FE.

    Returns
    -------
    numpy.ndarray
        The extracted overlay data as an array of 0s and 1s, shaped
        (rows, columns) for a single frame overlay or (frames, rows,
        columns) for a multi-frame overlay.

    Raises
    ------
    AttributeError
        If the dataset is missing a required element.
    ValueError
        If the overlay is already stored in (60xx,3000) *Overlay Data*, if
        the overlay bit falls within the stored pixel value bits or outside
        the allocated bits, if the overlay frames or plane are inconsistent
        with the image, or if the pixel data is compressed or otherwise
        unsupported.
    """
    if not HAVE_NP:
        raise ImportError("extract_overlay() requires numpy")

    if not 0x6000 <= group <= 0x60FE or group % 2:
        raise ValueError(
            f"The overlay group must be an even number between 0x6000 and "
            f"0x60FE, not 0x{group:04X}"
        )

    # Check the required elements are present
    elem = {
        "OverlayRows": ds.get((group, 0x0010), None),
        "OverlayColumns": ds.get((group, 0x0011), None),
        "OverlayBitsAllocated": ds.get((group, 0x0100), None),
        "OverlayBitPosition": ds.get((group, 0x0102), None),
        "PixelData": ds.get((0x7FE0, 0x0010), None),
        "Rows": ds.get((0x0028, 0x0010), None),
        "Columns": ds.get((0x0028, 0x0011), None),
        "BitsAllocated": ds.get((0x0028, 0x0100), None),
        "BitsStored": ds.get((0x0028, 0x0101), None),
        "PixelRepresentation": ds.get((0x0028, 0x0103), None),
        "SamplesPerPixel": ds.get((0x0028, 0x0002), None),
    }
    missing = [kk for kk, vv in elem.items() if vv is None]
    if missing:
        raise AttributeError(
            "Unable to extract the overlay data as the following required "
            f"elements are missing from the dataset: {', '.join(missing)}"
        )

    if (group, 0x3000) in ds:
        raise ValueError(
            f"Unable to extract the overlay data as (0x{group:04X},0x3000) "
            "'Overlay Data' is already present in the dataset"
        )

    # Only uncompressed transfer syntaxes contain usable native pixel data
    file_meta = getattr(ds, "file_meta", None)
    tsyntax = file_meta.get("TransferSyntaxUID", None) if file_meta else None
    if tsyntax is not None and tsyntax.is_compressed:
        raise ValueError(
            "Unable to extract the overlay data from pixel data encoded "
            f"using the compressed transfer syntax '{tsyntax}' ({tsyntax.name})"
        )

    # Only integer pixels with a single sample per pixel are supported
    if elem["SamplesPerPixel"].value != 1:
        raise ValueError(
            "Unable to extract the overlay data as only pixel data with "
            f"1 sample per pixel is supported, not "
            f"{elem['SamplesPerPixel'].value}"
        )

    rows = cast(int, elem["Rows"].value)
    columns = cast(int, elem["Columns"].value)
    bits_allocated = cast(int, elem["BitsAllocated"].value)
    bits_stored = cast(int, elem["BitsStored"].value)

    if not 1 <= bits_stored <= bits_allocated:
        raise ValueError(
            f"Unable to extract the overlay data as (0028,0101) 'Bits "
            f"Allocated' ({bits_allocated}) and (0028,0100) 'Bits Stored' "
            f"({bits_stored}) are inconsistent"
        )

    if elem["OverlayBitsAllocated"].value != bits_allocated:
        raise ValueError(
            f"Unable to extract the overlay data as (0x{group:04X},0x0100) "
            f"'Overlay Bits Allocated' ({elem['OverlayBitsAllocated'].value}) "
            f"doesn't match (0028,0101) 'Bits Allocated' ({bits_allocated}) "
            "as required for overlays stored in the pixel data"
        )

    bit_position = cast(int, elem["OverlayBitPosition"].value)
    if bit_position < bits_stored:
        raise ValueError(
            f"Unable to extract the overlay data as (0x{group:04X},0x0102) "
            f"'Overlay Bit Position' ({bit_position}) falls within the "
            f"stored pixel value bits (0 to {bits_stored - 1})"
        )
    if bit_position >= bits_allocated:
        raise ValueError(
            f"Unable to extract the overlay data as (0x{group:04X},0x0102) "
            f"'Overlay Bit Position' ({bit_position}) is outside the "
            f"allocated pixel bits (0 to {bits_allocated - 1})"
        )

    ov_rows = cast(int, elem["OverlayRows"].value)
    ov_columns = cast(int, elem["OverlayColumns"].value)
    if ov_rows < 1 or ov_columns < 1:
        raise ValueError(
            f"Unable to extract the overlay data as (0x{group:04X},0x0010) "
            f"'Overlay Rows' ({ov_rows}) and (0x{group:04X},0x0011) "
            f"'Overlay Columns' ({ov_columns}) must be greater than 0"
        )

    # Optional overlay elements
    origin_elem = ds.get((group, 0x0050), None)
    origin = [1, 1] if origin_elem is None else list(origin_elem.value)
    frame_origin_elem = ds.get((group, 0x0051), None)
    frame_origin = 1 if frame_origin_elem is None else int(frame_origin_elem.value)
    nr_frames_elem = ds.get((group, 0x0015), None)
    nr_ov_frames = 1 if nr_frames_elem is None else int(nr_frames_elem.value)

    nr_frames = 1
    if getattr(ds, "NumberOfFrames", None) not in (None, ""):
        nr_frames = int(ds.NumberOfFrames)

    if nr_ov_frames < 1 or not 1 <= frame_origin <= nr_frames:
        raise ValueError(
            f"Unable to extract the overlay data as (0x{group:04X},0x0051) "
            f"'Overlay Frame Origin' ({frame_origin}) and (0x{group:04X},"
            f"0x0015) 'Number of Frames in Overlay' ({nr_ov_frames}) are "
            f"inconsistent with the {nr_frames} frame(s) in the pixel data"
        )
    if frame_origin + nr_ov_frames - 1 > nr_frames:
        raise ValueError(
            f"Unable to extract the overlay data as the {nr_ov_frames} "
            f"overlay frame(s) starting at image frame {frame_origin} "
            f"exceed the {nr_frames} frame(s) in the pixel data"
        )

    # Determine the region where the overlay plane overlaps the image;
    #   overlay and image coordinates are 1-based
    origin_row, origin_column = int(origin[0]), int(origin[1])
    img_row = (max(origin_row - 1, 0), min(origin_row - 1 + ov_rows, rows))
    img_col = (max(origin_column - 1, 0), min(origin_column - 1 + ov_columns, columns))
    if img_row[0] >= img_row[1] or img_col[0] >= img_col[1]:
        raise ValueError(
            f"Unable to extract the overlay data as the overlay plane "
            f"starting at (0x{group:04X},0x0050) 'Overlay Origin' "
            f"[{origin_row}, {origin_column}] doesn't overlap the "
            f"{rows} x {columns} image"
        )

    # Read the pixel data as unsigned integers so the pixel value encoding
    #   (including two's complement signed values) is preserved
    dtype = pixel_dtype(ds)
    dtype = np.dtype(f"u{dtype.itemsize}").newbyteorder(dtype.byteorder)
    pixel_bytes = bytes(elem["PixelData"].value)
    expected = rows * columns * nr_frames * dtype.itemsize
    if len(pixel_bytes) < expected:
        raise ValueError(
            f"Unable to extract the overlay data as the length of the "
            f"pixel data ({len(pixel_bytes)} bytes) is less than the "
            f"expected length ({expected} bytes)"
        )
    if len(pixel_bytes) > expected + expected % 2:
        warn_and_log(
            f"The length of the pixel data ({len(pixel_bytes)} bytes) "
            "indicates it contains excess padding. The extra bytes at the "
            "end of the data will be left unchanged"
        )

    pixels = np.frombuffer(pixel_bytes[:expected], dtype=dtype)
    pixels = pixels.reshape(nr_frames, rows, columns)

    # Extract the overlay bits for the overlapping region
    overlay = np.zeros((nr_ov_frames, ov_rows, ov_columns), dtype=np.uint8)
    ov_row = (img_row[0] - (origin_row - 1), img_row[1] - (origin_row - 1))
    ov_col = (img_col[0] - (origin_column - 1), img_col[1] - (origin_column - 1))
    bits = (pixels >> bit_position) & 1
    overlay[:, ov_row[0] : ov_row[1], ov_col[0] : ov_col[1]] = bits[
        frame_origin - 1 : frame_origin - 1 + nr_ov_frames,
        img_row[0] : img_row[1],
        img_col[0] : img_col[1],
    ]

    # Clear only this overlay's bits from the frames the overlay applies to
    bit_mask = np.left_shift(np.array(1, dtype=dtype), bit_position)
    cleared = pixels.copy()
    cleared[frame_origin - 1 : frame_origin - 1 + nr_ov_frames] &= ~bit_mask
    cleared_bytes = cleared.tobytes()
    ds.PixelData = cleared_bytes + pixel_bytes[len(cleared_bytes) :]

    # Convert the overlay plane to a standalone Overlay Data element
    ds[Tag(group, 0x0100)].value = 1  # Overlay Bits Allocated
    ds[Tag(group, 0x0102)].value = 0  # Overlay Bit Position
    ds.add(DataElement((group, 0x3000), "OW", pack_bits(overlay)))

    if nr_ov_frames == 1:
        return overlay.reshape(ov_rows, ov_columns)

    return overlay
