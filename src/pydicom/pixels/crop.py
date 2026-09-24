# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Crop rectangular regions from uncompressed grayscale pixel data."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

from pydicom.pixels.utils import pixel_dtype

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


def _fg_value(item: "Dataset | None", seq_keyword: str, elem_keyword: str) -> Any:
    """Return the value of 'elem_keyword' from the first item of the
    'seq_keyword' sequence in 'item', or None if not present."""
    if item is None:
        return None

    seq = item.get(seq_keyword)
    if seq:
        return seq[0].get(elem_keyword)

    return None


def _validate_orientation(
    value: Any, frame_index: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Validate (0020,0037) *Image Orientation Patient* 'value' and return
    the row and column direction cosines as unit vectors."""
    cosines = [float(v) for v in value]
    if len(cosines) != 6:
        raise ValueError(
            f"Unsupported geometry for frame {frame_index}: (0020,0037) 'Image "
            f"Orientation Patient' must have 6 values, got {len(cosines)}"
        )

    row_cosines = np.array(cosines[:3])
    col_cosines = np.array(cosines[3:])
    if not np.isclose(np.dot(row_cosines, row_cosines), 1.0, atol=1e-3):
        raise ValueError(
            f"Unsupported geometry for frame {frame_index}: the row direction "
            "cosines in (0020,0037) 'Image Orientation Patient' are not a unit "
            "vector"
        )
    if not np.isclose(np.dot(col_cosines, col_cosines), 1.0, atol=1e-3):
        raise ValueError(
            f"Unsupported geometry for frame {frame_index}: the column "
            "direction cosines in (0020,0037) 'Image Orientation Patient' are "
            "not a unit vector"
        )
    if not np.isclose(np.dot(row_cosines, col_cosines), 0.0, atol=1e-3):
        raise ValueError(
            f"Unsupported geometry for frame {frame_index}: the row and column "
            "direction cosines in (0020,0037) 'Image Orientation Patient' are "
            "not orthogonal"
        )

    return row_cosines, col_cosines


def _validate_spacing(value: Any, frame_index: int) -> tuple[float, float]:
    """Validate (0028,0030) *Pixel Spacing* 'value' and return the
    (row spacing, column spacing) as floats."""
    spacing = [float(v) for v in value]
    if len(spacing) != 2 or spacing[0] <= 0 or spacing[1] <= 0:
        raise ValueError(
            f"Unsupported geometry for frame {frame_index}: (0028,0030) 'Pixel "
            "Spacing' must contain two positive values"
        )

    return spacing[0], spacing[1]


def crop_pixels(ds: "Dataset", crop_box: tuple[int, int, int, int]) -> "Dataset":
    """Crop a rectangular region from the pixel data in 'ds' and return a
    new Dataset with correctly updated spatial information.

    .. versionadded:: 3.1

    Supports uncompressed (native) single frame and Enhanced multi-frame
    grayscale pixel data. The crop is performed on whole pixels only, no
    interpolation is applied. The original dataset 'ds' is left unchanged.

    The following elements are updated in the returned dataset:

    * (0028,0010) *Rows* and (0028,0011) *Columns* from 'crop_box'.
    * (7FE0,0010) *Pixel Data* with the cropped frames.
    * (0020,0032) *Image Position Patient*, translated along the row and
      column direction cosines of (0020,0037) *Image Orientation Patient* by
      the crop origin scaled with (0028,0030) *Pixel Spacing*. For Enhanced
      multi-frame data the position of every frame is updated individually
      within its own *Per-frame Functional Groups Sequence* item; values from
      the *Shared Functional Groups Sequence* are only used when a frame has
      no per-frame value.

    All other elements, including (0028,0100) *Bits Allocated*, (0028,0101)
    *Bits Stored*, (0028,0102) *High Bit* and (0028,0103) *Pixel
    Representation*, are preserved unchanged.

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        A dataset containing uncompressed grayscale pixel data and the
        corresponding spatial information.
    crop_box : tuple[int, int, int, int]
        The crop region as '(top, left, height, width)' in pixel indices,
        where '(top, left)' is the row and column index of the first kept
        pixel. The region must lie fully within the image.

    Returns
    -------
    pydicom.dataset.Dataset
        A new dataset containing the cropped pixel data.

    Raises
    ------
    NotImplementedError
        If the pixel data is encapsulated (compressed), contains more than
        one sample per pixel, or is bit-packed.
    ValueError
        If 'crop_box' is invalid or lies outside the image, if a required
        element is missing, or if the spatial geometry is missing or
        unsupported (e.g. non-orthogonal direction cosines). The original
        dataset is not modified when an exception is raised.
    """
    if not HAVE_NP:
        raise ImportError("Numpy is required to crop pixel data")

    try:
        top, left, height, width = crop_box
    except (TypeError, ValueError):
        raise ValueError(
            "'crop_box' must be a (top, left, height, width) tuple of ints"
        ) from None

    for name, value in zip(
        ("top", "left", "height", "width"), (top, left, height, width)
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"'crop_box' values must be ints, got {value!r} for '{name}'"
            )

    top, left, height, width = int(top), int(left), int(height), int(width)

    for keyword in (
        "PixelData",
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "PixelRepresentation",
    ):
        if keyword not in ds:
            raise ValueError(
                f"Unable to crop the dataset, required element '{keyword}' is missing"
            )

    file_meta = getattr(ds, "file_meta", None)
    tsyntax = file_meta.get("TransferSyntaxUID") if file_meta else None
    if tsyntax is not None and tsyntax.is_encapsulated:
        raise NotImplementedError(
            f"Cropping pixel data compressed with '{tsyntax.name}' is not "
            "supported, only uncompressed (native) pixel data can be cropped"
        )

    if int(ds.get("SamplesPerPixel", 1)) != 1:
        raise NotImplementedError(
            "Cropping pixel data with more than one sample per pixel (such as "
            "RGB) is not supported, only grayscale data can be cropped"
        )

    bits_allocated = int(ds.BitsAllocated)
    if bits_allocated == 1 or bits_allocated % 8 != 0:
        raise NotImplementedError(
            f"Cropping bit-packed pixel data ('Bits Allocated' is "
            f"{bits_allocated}) is not supported"
        )

    rows, columns = int(ds.Rows), int(ds.Columns)
    if height < 1 or width < 1:
        raise ValueError(
            f"The crop box height and width must be >= 1, got ({height}, {width})"
        )
    if top < 0 or left < 0 or top + height > rows or left + width > columns:
        raise ValueError(
            f"The crop box (top={top}, left={left}, height={height}, "
            f"width={width}) lies outside the {rows} x {columns} image"
        )

    # Only the copy is modified from here on, the original stays untouched
    new = deepcopy(ds)

    nr_frames = int(getattr(new, "NumberOfFrames", 1) or 1)

    shared = None
    if "SharedFunctionalGroupsSequence" in new:
        shared = new.SharedFunctionalGroupsSequence[0]
    per_frame = new.get("PerFrameFunctionalGroupsSequence")
    if per_frame is not None and len(per_frame) != nr_frames:
        raise ValueError(
            f"The number of items ({len(per_frame)}) in (5200,9230) 'Per-frame "
            f"Functional Groups Sequence' doesn't match (0028,0008) 'Number of "
            f"Frames' ({nr_frames})"
        )

    # Resolve and validate the geometry for every frame within its own scope:
    # per-frame functional groups take precedence over shared functional
    # groups, which take precedence over top-level elements
    updates = []
    for frame_index in range(nr_frames):
        frame_item = per_frame[frame_index] if per_frame is not None else None

        spacing = _fg_value(frame_item, "PixelMeasuresSequence", "PixelSpacing")
        if spacing is None:
            spacing = _fg_value(shared, "PixelMeasuresSequence", "PixelSpacing")
        if spacing is None:
            spacing = new.get("PixelSpacing")
        if spacing is None:
            raise ValueError(
                f"Unable to crop frame {frame_index}: (0028,0030) 'Pixel "
                "Spacing' is missing from the per-frame and shared functional "
                "groups and from the top-level dataset"
            )
        row_spacing, col_spacing = _validate_spacing(spacing, frame_index)

        orientation = _fg_value(
            frame_item, "PlaneOrientationSequence", "ImageOrientationPatient"
        )
        if orientation is None:
            orientation = _fg_value(
                shared, "PlaneOrientationSequence", "ImageOrientationPatient"
            )
        if orientation is None:
            orientation = new.get("ImageOrientationPatient")
        if orientation is None:
            raise ValueError(
                f"Unable to crop frame {frame_index}: (0020,0037) 'Image "
                "Orientation Patient' is missing from the per-frame and shared "
                "functional groups and from the top-level dataset"
            )
        row_cosines, col_cosines = _validate_orientation(orientation, frame_index)

        holder = new
        position = None
        for scope in (frame_item, shared):
            if scope is not None and "PlanePositionSequence" in scope:
                candidate = scope.PlanePositionSequence[0]
                position = candidate.get("ImagePositionPatient")
                if position is not None:
                    holder = candidate
                    break
        if position is None:
            holder = new
            position = new.get("ImagePositionPatient")
        if position is None:
            raise ValueError(
                f"Unable to crop frame {frame_index}: (0020,0032) 'Image "
                "Position Patient' is missing from the per-frame functional "
                "groups and from the top-level dataset"
            )

        origin = np.array([float(v) for v in position])
        offset = row_cosines * (top * row_spacing) + col_cosines * (left * col_spacing)
        updates.append((holder, list(origin + offset)))

    # Decode and crop the pixel data without changing the sample dtype
    dtype = pixel_dtype(new)
    arr = np.frombuffer(new.PixelData, dtype=dtype)
    expected = nr_frames * rows * columns
    if arr.size < expected:
        raise ValueError(
            f"The length of (7FE0,0010) 'Pixel Data' ({arr.size} pixels) is "
            f"less than expected for {nr_frames} frame(s) of {rows} x "
            f"{columns} pixels"
        )

    arr = arr[:expected].reshape(nr_frames, rows, columns)
    cropped = arr[:, top : top + height, left : left + width]

    new.Rows = height
    new.Columns = width
    # Update the value of the existing element to preserve its VR
    new["PixelData"].value = cropped.tobytes()

    for holder, position in updates:
        holder.ImagePositionPatient = position

    return new
