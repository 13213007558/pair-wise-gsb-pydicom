# Copyright 2008-2024 pydicom authors. See LICENSE file for details.
"""Crop rectangular regions from uncompressed grayscale pixel data."""

import copy
import math
from typing import TYPE_CHECKING

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

from pydicom.pixels.utils import pixel_dtype
from pydicom.uid import UID

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


# Tolerance used when validating the direction cosines
_COSINE_TOLERANCE = 1e-3


def _require(ds: "Dataset", keyword: str) -> object:
    """Return the value of the element 'keyword' or raise ValueError."""
    value = ds.get(keyword)
    if value is None:
        raise ValueError(
            f"Unable to crop the pixel data: the dataset is missing the "
            f"required '{keyword}' element"
        )
    return value


def _sequence_value(ds: "Dataset", seq_keyword: str, keyword: str) -> object:
    """Return the value of 'keyword' from the first item of 'seq_keyword'."""
    seq = ds.get(seq_keyword)
    if seq is None or len(seq) == 0:
        return None
    return seq[0].get(keyword)


def _check_orientation(iop: object) -> tuple[list[float], list[float]]:
    """Validate *Image Orientation Patient* and return the direction cosines.

    Returns a '(row cosines, column cosines)' tuple of unit vectors.
    """
    if iop is None or len(iop) != 6:  # type: ignore[arg-type]
        raise ValueError(
            "Unable to crop the pixel data: cropping requires a 6-value "
            "'ImageOrientationPatient' element"
        )

    row_cosines = [float(v) for v in iop[:3]]  # type: ignore[index]
    column_cosines = [float(v) for v in iop[3:]]  # type: ignore[index]

    for name, cosines in (
        ("row", row_cosines),
        ("column", column_cosines),
    ):
        norm = math.sqrt(sum(v * v for v in cosines))
        if not math.isclose(norm, 1.0, abs_tol=_COSINE_TOLERANCE):
            raise ValueError(
                "Unable to crop the pixel data: unsupported image geometry, "
                f"the {name} direction cosines {cosines} are not a unit vector"
            )

    dot = sum(r * c for r, c in zip(row_cosines, column_cosines))
    if not math.isclose(dot, 0.0, abs_tol=_COSINE_TOLERANCE):
        raise ValueError(
            "Unable to crop the pixel data: unsupported image geometry, the "
            "row and column direction cosines are not orthogonal"
        )

    return row_cosines, column_cosines


def _check_spacing(spacing: object) -> tuple[float, float]:
    """Validate *Pixel Spacing* and return '(row spacing, column spacing)'."""
    if spacing is None or len(spacing) != 2:  # type: ignore[arg-type]
        raise ValueError(
            "Unable to crop the pixel data: cropping requires a 2-value "
            "'PixelSpacing' element"
        )

    row_spacing, column_spacing = (float(v) for v in spacing)  # type: ignore[union-attr]
    if row_spacing <= 0 or column_spacing <= 0:
        raise ValueError(
            "Unable to crop the pixel data: 'PixelSpacing' values must be "
            f"greater than zero, got {list(spacing)}"  # type: ignore[arg-type]
        )

    return row_spacing, column_spacing


def _check_position(ipp: object, scope: str) -> list[float]:
    """Validate *Image Position Patient* and return it as a list of floats."""
    if ipp is None or len(ipp) != 3:  # type: ignore[arg-type]
        raise ValueError(
            "Unable to crop the pixel data: cropping requires a 3-value "
            f"'ImagePositionPatient' element in the {scope}"
        )
    return [float(v) for v in ipp]  # type: ignore[union-attr]


def _frame_geometries(
    ds: "Dataset", nr_frames: int
) -> list[tuple[list[float], tuple[list[float], list[float]], tuple[float, float]]]:
    """Return the '(position, cosines, spacing)' geometry for each frame.

    The geometry is returned per frame, using the shared and per-frame
    functional groups according to their respective scopes for enhanced
    multi-frame images, or the top-level elements otherwise.
    """
    if "PerFrameFunctionalGroupsSequence" not in ds:
        # Single frame image using the top-level Image Plane module
        ipp = _check_position(ds.get("ImagePositionPatient"), "dataset")
        cosines = _check_orientation(ds.get("ImageOrientationPatient"))
        spacing = _check_spacing(ds.get("PixelSpacing"))
        return [(ipp, cosines, spacing)]

    # Enhanced multi-frame image
    shared = ds.get("SharedFunctionalGroupsSequence")
    shared_item = shared[0] if shared is not None and len(shared) else None

    per_frame = ds.PerFrameFunctionalGroupsSequence
    if len(per_frame) != nr_frames:
        raise ValueError(
            "Unable to crop the pixel data: the number of items in "
            f"'PerFrameFunctionalGroupsSequence' ({len(per_frame)}) doesn't "
            f"match the value of 'NumberOfFrames' ({nr_frames})"
        )

    geometries = []
    for index, item in enumerate(per_frame):
        # The patient position is inherently per-frame and must not be
        #   overridden by a single top-level value
        ipp = _check_position(
            _sequence_value(item, "PlanePositionSequence", "ImagePositionPatient"),
            f"per-frame functional groups item {index}",
        )

        iop = _sequence_value(
            item, "PlaneOrientationSequence", "ImageOrientationPatient"
        )
        if iop is None and shared_item is not None:
            iop = _sequence_value(
                shared_item, "PlaneOrientationSequence", "ImageOrientationPatient"
            )
        cosines = _check_orientation(iop)

        spacing = _sequence_value(item, "PixelMeasuresSequence", "PixelSpacing")
        if spacing is None and shared_item is not None:
            spacing = _sequence_value(
                shared_item, "PixelMeasuresSequence", "PixelSpacing"
            )
        spacings = _check_spacing(spacing)

        geometries.append((ipp, cosines, spacings))

    return geometries


def _translate_position(
    ipp: list[float],
    cosines: tuple[list[float], list[float]],
    spacing: tuple[float, float],
    left: int,
    upper: int,
) -> list[float]:
    """Return the *Image Position Patient* for the given crop origin."""
    row_cosines, column_cosines = cosines
    row_spacing, column_spacing = spacing
    return [
        origin + row_cos * left * column_spacing + column_cos * upper * row_spacing
        for origin, row_cos, column_cos in zip(ipp, row_cosines, column_cosines)
    ]


def crop(ds: "Dataset", box: tuple[int, int, int, int]) -> "Dataset":
    """Return a new Dataset with the pixel data cropped to a rectangle.

    The first version supports uncompressed single frame and enhanced
    multi-frame grayscale images. The pixels are copied without interpolation,
    so the bit depth, signedness and *Bits Stored* of the original data are
    preserved. The *Rows*, *Columns* and *Pixel Data* elements are updated,
    and the *Image Position Patient* of every frame is translated along the
    row and column direction cosines by the corresponding *Pixel Spacing* so
    the cropped image keeps its correct location in the patient coordinate
    system. For enhanced multi-frame images the shared and per-frame
    functional groups are used according to their respective scopes.

    Parameters
    ----------
    ds : Dataset
        The dataset to crop. Required elements are *Rows*, *Columns*,
        *Pixel Data*, *Pixel Spacing*, *Image Position Patient* and
        *Image Orientation Patient* (or their functional group equivalents
        for enhanced multi-frame images).
    box : tuple of int
        The crop rectangle as '(left, upper, right, lower)' in pixel
        coordinates, where 'left'/'upper' are inclusive and 'right'/'lower'
        are exclusive.

    Returns
    -------
    Dataset
        A new dataset containing the cropped pixel data. The original
        dataset is left unchanged.

    Raises
    ------
    ImportError
        If numpy is not available.
    ValueError
        If the crop box is out of bounds, the required spatial elements are
        missing or invalid, or the image uses a geometry or encoding that is
        not supported. The original dataset is left unchanged.
    """
    if not HAVE_NP:
        raise ImportError("Numpy is required to crop pixel data")

    # Validate everything against the original dataset first so a failure
    #   leaves 'ds' untouched
    try:
        left, upper, right, lower = (int(v) for v in box)
    except (TypeError, ValueError):
        raise ValueError(
            "'box' must be a (left, upper, right, lower) tuple of 4 integers"
        ) from None

    rows = int(_require(ds, "Rows"))
    columns = int(_require(ds, "Columns"))
    if not (0 <= left < right <= columns) or not (0 <= upper < lower <= rows):
        raise ValueError(
            f"Unable to crop the pixel data: the crop box ({left}, {upper}, "
            f"{right}, {lower}) is out of bounds for a {rows} x {columns} image"
        )

    if "PixelData" not in ds:
        raise ValueError(
            "Unable to crop the pixel data: the dataset contains no 'Pixel "
            "Data' element"
        )

    file_meta = getattr(ds, "file_meta", None)
    if file_meta is not None and "TransferSyntaxUID" in file_meta:
        uid = UID(file_meta.TransferSyntaxUID)
        if uid.is_compressed:
            raise ValueError(
                "Unable to crop the pixel data: compressed transfer syntaxes "
                f"such as '{uid.name}' are not supported"
            )

    if int(ds.get("SamplesPerPixel", 1)) != 1:
        raise ValueError(
            "Unable to crop the pixel data: only grayscale images with "
            f"'SamplesPerPixel' of 1 are supported, got "
            f"{int(ds.SamplesPerPixel)}"
        )

    photometric = ds.get("PhotometricInterpretation")
    if photometric is not None and photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(
            "Unable to crop the pixel data: only grayscale images with a "
            "'PhotometricInterpretation' of MONOCHROME1 or MONOCHROME2 are "
            f"supported, got '{photometric}'"
        )

    bits_allocated = int(_require(ds, "BitsAllocated"))
    if bits_allocated == 1:
        raise ValueError(
            "Unable to crop the pixel data: bit-packed pixel data with "
            "'BitsAllocated' of 1 is not supported"
        )

    nr_frames = int(ds.get("NumberOfFrames", 1))
    expected = nr_frames * rows * columns * (bits_allocated // 8)
    if len(ds.PixelData) != expected:
        raise ValueError(
            "Unable to crop the pixel data: the length of 'Pixel Data' "
            f"({len(ds.PixelData)} bytes) doesn't match the expected length "
            f"of {expected} bytes for uncompressed pixel data"
        )

    try:
        dtype = pixel_dtype(ds)
    except AttributeError as err:
        raise ValueError(
            "Unable to crop the pixel data: unable to determine the encoding "
            "of the dataset, please set an appropriate Transfer Syntax UID "
            "in the dataset's file meta information"
        ) from err

    geometries = _frame_geometries(ds, nr_frames)

    # All validation passed - apply the crop to a copy of the dataset
    new = copy.deepcopy(ds)

    arr = np.frombuffer(ds.PixelData, dtype=dtype)
    if nr_frames > 1:
        arr = arr.reshape(nr_frames, rows, columns)
    else:
        arr = arr.reshape(rows, columns)
    new.PixelData = arr[..., upper:lower, left:right].tobytes()

    new.Rows = lower - upper
    new.Columns = right - left

    if "PerFrameFunctionalGroupsSequence" in new:
        for item, (ipp, cosines, spacing) in zip(
            new.PerFrameFunctionalGroupsSequence, geometries
        ):
            item.PlanePositionSequence[0].ImagePositionPatient = _translate_position(
                ipp, cosines, spacing, left, upper
            )
    else:
        ipp, cosines, spacing = geometries[0]
        new.ImagePositionPatient = _translate_position(
            ipp, cosines, spacing, left, upper
        )

    return new
