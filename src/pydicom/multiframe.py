"""Functions for working with enhanced multi-frame datasets."""

from collections.abc import Callable, Iterable
import copy
from io import BytesIO
import operator
from typing import TYPE_CHECKING

from pydicom.encaps import (
    encapsulate,
    encapsulate_extended,
    generate_frames,
    parse_basic_offsets,
)
from pydicom.multival import MultiValue
from pydicom.pixels.utils import concatenate_packed_frames, get_packed_frame
from pydicom.sequence import Sequence
from pydicom.uid import UID, generate_uid

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


# (0008,1160) Referenced Frame Number
REFERENCED_FRAME_NUMBER = 0x00081160


def select_frames(
    ds: "Dataset",
    indices: Iterable[int],
    *,
    sop_instance_uid: "str | UID | Callable[[Dataset], str | UID] | None" = None,
) -> "Dataset":
    """Return a new dataset containing only the frames given by `indices`.

    .. versionadded:: 3.1

    This is intended for use with enhanced multi-frame datasets such as
    *Enhanced MR Image Storage* and performs a dataset-level frame
    selection rather than simply slicing the *Pixel Data*:

    * The order and any duplicates in `indices` are preserved, so
      ``select_frames(ds, [4, 1, 4])`` returns a dataset whose first and
      third frames are both the frame at index `` 4 `` of `ds`.
    * (0028,0008) *Number of Frames* is updated to match.
    * (5200,9230) *Per-frame Functional Groups Sequence* is reordered and
      duplicated to correspond item-by-item with the selected frames, while
      (5200,9229) *Shared Functional Groups Sequence* is kept unchanged so
      the shared semantics are preserved.
    * (0008,1160) *Referenced Frame Number* values anywhere in the dataset
      are remapped to the new frame positions. A frame number that is
      selected more than once is mapped to its first occurrence, and
      references to frames that are not selected are removed.
    * (0008,0018) *SOP Instance UID* of the output is produced by the
      caller-supplied `sop_instance_uid` policy (or a new UID is generated
      if no policy is given). (0002,0003) *Media Storage SOP Instance UID*
      in the file meta information, if present, is updated to match.

    For natively encoded (uncompressed) pixel data the frames are extracted
    directly, including bit-packed single bit data (*Bits Allocated* of 1)
    where a single byte may contain pixels from more than one frame. For
    encapsulated (compressed) pixel data the compressed frames are copied
    without decoding and the Basic or Extended Offset Table is rebuilt to
    match the new frame set. If the frame boundaries cannot be determined
    reliably (the Basic Offset Table is empty or incomplete and no Extended
    Offset Table is present) a :class:`ValueError` is raised.

    The input dataset `ds` is left unchanged.

    Parameters
    ----------
    ds : Dataset
        The multi-frame dataset to select frames from. Must contain
        (7FE0,0010) *Pixel Data* and the
        :dcm:`Image Pixel<part03/sect_C.7.6.3.html>` module elements
        required to interpret it.
    indices : Iterable[int]
        The zero-based indices of the frames to select, in the desired
        order. Indices may be repeated and negative indices count back from
        the last frame in the usual Python manner.
    sop_instance_uid : str, UID, callable or None, optional
        The policy used to produce the *SOP Instance UID* of the output
        dataset. If a :class:`str` or :class:`~pydicom.uid.UID` then it
        is used directly. If callable then it will be called with the new
        dataset (after frame selection) and must return the UID to use. If
        ``None`` (default) then a new UID is generated using
        :func:`~pydicom.uid.generate_uid`.

    Returns
    -------
    Dataset
        A new dataset containing only the selected frames.

    Raises
    ------
    IndexError
        If an index is out of range for the number of frames in `ds`.
    ValueError
        If `indices` is empty, if the dataset is inconsistent (for example
        the *Per-frame Functional Groups Sequence* length doesn't match
        *Number of Frames*), or if the boundaries of encapsulated frames
        cannot be determined reliably.
    """
    indices = [operator.index(idx) for idx in indices]
    if not indices:
        raise ValueError("'indices' must contain at least one frame index")

    if "PixelData" not in ds:
        raise ValueError("The dataset contains no 'Pixel Data' element")

    nr_frames = int(getattr(ds, "NumberOfFrames", 1))
    if nr_frames < 1:
        raise ValueError("The dataset contains no frames")

    normalized = []
    for idx in indices:
        frame_index = idx + nr_frames if idx < 0 else idx
        if not 0 <= frame_index < nr_frames:
            raise IndexError(
                f"Frame index {idx} is out of range for a dataset with "
                f"{nr_frames} frames"
            )
        normalized.append(frame_index)

    if "PerFrameFunctionalGroupsSequence" in ds:
        nr_items = len(ds.PerFrameFunctionalGroupsSequence)
        if nr_items != nr_frames:
            raise ValueError(
                f"The 'Per-frame Functional Groups Sequence' contains "
                f"{nr_items} items but 'Number of Frames' is {nr_frames}; "
                f"unable to select frames from an inconsistent dataset"
            )

    out = copy.deepcopy(ds)

    if _is_encapsulated(out):
        _select_encapsulated_frames(out, normalized, nr_frames)
    else:
        _select_native_frames(out, normalized, nr_frames)

    out.NumberOfFrames = len(normalized)

    if "PerFrameFunctionalGroupsSequence" in out:
        per_frame = out.PerFrameFunctionalGroupsSequence
        out.PerFrameFunctionalGroupsSequence = Sequence(
            # Copy each item so duplicated frames get independent items
            [copy.deepcopy(per_frame[ii]) for ii in normalized]
        )

    # Remap (0008,1160) Referenced Frame Number to the new frame positions;
    # a frame selected more than once is referenced by its first occurrence
    frame_mapping: dict[int, int] = {}
    for new_index, old_index in enumerate(normalized):
        frame_mapping.setdefault(old_index + 1, new_index + 1)
    _remap_frame_references(out, frame_mapping)

    # Clear any cached pixel data as it no longer matches
    out._pixel_array = None
    out._pixel_id = {}

    # The output SOP Instance UID is produced by the caller's policy
    if sop_instance_uid is None:
        new_uid = generate_uid()
    elif callable(sop_instance_uid):
        new_uid = UID(str(sop_instance_uid(out)))
    else:
        new_uid = UID(str(sop_instance_uid))

    out.SOPInstanceUID = new_uid
    file_meta = getattr(out, "file_meta", None)
    if file_meta is not None and "MediaStorageSOPInstanceUID" in file_meta:
        file_meta.MediaStorageSOPInstanceUID = new_uid

    return out


def _is_encapsulated(ds: "Dataset") -> bool:
    """Return ``True`` if the dataset's pixel data is encapsulated."""
    elem = ds["PixelData"]
    if elem.is_undefined_length:
        return True

    file_meta = getattr(ds, "file_meta", None)
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)

    return bool(transfer_syntax is not None and transfer_syntax.is_encapsulated)


def _select_native_frames(ds: "Dataset", indices: list[int], nr_frames: int) -> None:
    """Update the native (uncompressed) *Pixel Data* in `ds` in-place."""
    rows = int(ds.Rows)
    columns = int(ds.Columns)
    samples_per_pixel = int(ds.SamplesPerPixel)
    bits_allocated = int(ds.BitsAllocated)
    frame_pixels = rows * columns * samples_per_pixel

    elem = ds["PixelData"]
    src = bytes(elem.value)

    if bits_allocated == 1:
        # Bit-packed pixel data: a single byte may contain pixels from
        #   more than one frame, so frames can't simply be sliced out
        frames = [
            get_packed_frame(src, index, frame_pixels, pad=False) for index in indices
        ]
        elem.value = concatenate_packed_frames(frames, frame_pixels, pad=True)
        return

    if bits_allocated % 8 != 0:
        raise ValueError(
            f"Unable to select frames from pixel data with 'Bits Allocated' "
            f"of {bits_allocated}; only 1 or a multiple of 8 is supported"
        )

    frame_length = frame_pixels * (bits_allocated // 8)
    if len(src) < frame_length * nr_frames:
        raise ValueError(
            f"The length of the 'Pixel Data' ({len(src)} bytes) is less than "
            f"the expected length of {frame_length * nr_frames} bytes for "
            f"{nr_frames} frames"
        )

    elem.value = b"".join(
        src[index * frame_length : (index + 1) * frame_length] for index in indices
    )
    if len(elem.value) % 2:
        elem.value += b"\x00"


def _select_encapsulated_frames(
    ds: "Dataset", indices: list[int], nr_frames: int
) -> None:
    """Update the encapsulated (compressed) *Pixel Data* in `ds` in-place.

    The compressed frames are copied without being decoded and the Basic or
    Extended Offset Table is rebuilt for the selected frames.
    """
    elem = ds["PixelData"]
    value = elem.value
    if isinstance(value, bytes | bytearray):
        src = bytes(value)
    elif hasattr(value, "read"):
        value.seek(0)
        src = value.read()
    else:
        raise TypeError(
            f"Unsupported 'Pixel Data' value type {type(value).__name__}; "
            f"expected bytes or a readable buffer"
        )

    extended_offsets = None
    if "ExtendedOffsetTable" in ds and "ExtendedOffsetTableLengths" in ds:
        extended_offsets = (
            bytes(ds.ExtendedOffsetTable),
            bytes(ds.ExtendedOffsetTableLengths),
        )

    if extended_offsets is not None:
        frames = list(
            generate_frames(
                src,
                number_of_frames=nr_frames,
                extended_offsets=extended_offsets,
            )
        )
        if len(frames) != nr_frames:
            raise ValueError(
                f"The Extended Offset Table yields {len(frames)} frames but "
                f"'Number of Frames' is {nr_frames}; unable to reliably "
                f"determine the encapsulated frame boundaries"
            )

        selected = [frames[ii] for ii in indices]
        (
            elem.value,
            ds.ExtendedOffsetTable,
            ds.ExtendedOffsetTableLengths,
        ) = encapsulate_extended(selected)
    else:
        offsets = parse_basic_offsets(BytesIO(src))
        if len(offsets) != nr_frames:
            raise ValueError(
                "Unable to reliably determine the encapsulated frame "
                "boundaries: the Basic Offset Table is empty or doesn't "
                "contain one offset per frame and no Extended Offset Table "
                "is present"
            )

        frames = list(generate_frames(src, number_of_frames=nr_frames))
        selected = [frames[ii] for ii in indices]
        elem.value = encapsulate(selected)

    # PS3.5 Annex A.4 - encapsulated pixel data uses undefined length and OB
    elem.is_undefined_length = True
    elem.VR = "OB"


def _remap_frame_references(ds: "Dataset", frame_mapping: dict[int, int]) -> None:
    """Remap (0008,1160) *Referenced Frame Number* values in `ds` in-place.

    Each referenced frame number is replaced by the new position of the
    referenced frame given by `frame_mapping`, which maps the original
    one-based frame numbers to the new one-based frame numbers. References
    to frames that aren't in `frame_mapping` are removed; elements left
    with no values are deleted.
    """
    empty_tags = []
    for elem in ds:
        if elem.tag == REFERENCED_FRAME_NUMBER:
            values = (
                list(elem.value)
                if isinstance(elem.value, MultiValue | list)
                else [elem.value]
            )
            remapped: list[int] = []
            for value in values:
                new_number = frame_mapping.get(int(value))
                if new_number is not None and new_number not in remapped:
                    remapped.append(new_number)

            if remapped:
                elem.value = remapped[0] if len(remapped) == 1 else remapped
            else:
                empty_tags.append(elem.tag)
        elif elem.VR == "SQ":
            for item in elem.value:
                _remap_frame_references(item, frame_mapping)

    for tag in empty_tags:
        del ds[tag]
