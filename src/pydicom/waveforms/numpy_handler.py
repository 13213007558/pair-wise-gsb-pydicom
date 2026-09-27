# Copyright 2008-2020 pydicom authors. See LICENSE file for details.
"""Use the `numpy <https://numpy.org/>`_ package to convert supported *Waveform
Data* to a :class:`numpy.ndarray`.

.. versionadded:: 2.1

**Supported data**

The numpy handler supports the conversion of data in the (5400,0100)
*Waveform Sequence* element to an :class:`~numpy.ndarray` provided the
related :dcm:`Waveform<part03/sect_C.10.9.html>` module elements have values
given in the table below.

+-------------+------------------------------+------+-------------------------+
| Element                                           | Supported               |
+-------------+------------------------------+------+ values                  |
| Tag         | Keyword                      | Type |                         |
+=============+==============================+======+=========================+
| (003A,0005) | NumberOfWaveformChannels     | 1    | N > 0                   |
+-------------+------------------------------+------+-------------------------+
| (003A,0010) | NumberOfWaveformSamples      | 1    | N > 0                   |
+-------------+------------------------------+------+-------------------------+
| (5400,1004) | WaveformBitsAllocated        | 1    | 8, 16, 32, 64           |
+-------------+------------------------------+------+-------------------------+
| (5400,1006) | WaveformSampleInterpretation | 1    | SB, UB, MB, AB, SS, US, |
|             |                              |      | SL, UL, SV, UV          |
+-------------+------------------------------+------+-------------------------+

"""

import math
import os
from typing import TYPE_CHECKING, Any, BinaryIO, NamedTuple, cast
from collections.abc import Generator, Sequence

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset, FileDataset


HANDLER_NAME = "Numpy Waveform"
DEPENDENCIES = {"numpy": ("https://numpy.org/", "NumPy")}
WAVEFORM_DTYPES = {
    (8, "SB"): "int8",
    (8, "UB"): "uint8",
    (8, "MB"): "uint8",
    (8, "AB"): "uint8",
    (16, "SS"): "<i2",
    (16, "US"): "<u2",
    (32, "SL"): "<i4",
    (32, "UL"): "<u4",
    (64, "SV"): "<i8",
    (64, "UV"): "<u8",
}


def is_available() -> bool:
    """Return ``True`` if the handler has its dependencies met.

    .. versionadded:: 2.1
    """
    return HAVE_NP


class WaveformWindow(NamedTuple):
    """Result returned by :func:`waveform_window`.

    .. versionadded:: 3.1
    """

    times: "np.ndarray"
    """The time axis in seconds, with shape `(samples,)` if all selected
    channels share the same skew, otherwise `(samples, channels)`."""

    samples: "np.ndarray"
    """The waveform samples with shape `(samples, channels)`, in the same
    channel order as requested."""


def _tolerant_ceil(value: float) -> int:
    """Return `ceil(value)`, snapping to the nearest integer when `value`
    is within floating point rounding error of it.
    """
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-9, abs_tol=1e-9):
        return int(nearest)

    return math.ceil(value)


def _window_indices(
    start: float, end: float, origin: float, frequency: float, nr_samples: int
) -> tuple[int, int]:
    """Return the `[first, last)` sample indices for the half-open time
    window `[start, end)` given samples at `origin + i / frequency`.
    """
    first = min(max(_tolerant_ceil((start - origin) * frequency), 0), nr_samples)
    last = min(max(_tolerant_ceil((end - origin) * frequency), 0), nr_samples)
    return first, max(first, last)


def generate_multiplex(
    ds: "Dataset", as_raw: bool = True
) -> Generator["np.ndarray", None, None]:
    """Yield an :class:`~numpy.ndarray` for each multiplex group in the
    *Waveform Sequence*.

    .. versionadded:: 2.1

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The :class:`Dataset` containing a :dcm:`Waveform
        <part03/sect_C.10.9.html>` module and the *Waveform Sequence* to be
        converted.
    as_raw : bool, optional
        If ``True`` (default), then yield the raw unitless waveform data. If
        ``False`` then attempt to convert the raw data for each channel to the
        quantity specified by the corresponding (003A,0210) *Channel
        Sensitivity* unit.

    Yields
    ------
    np.ndarray
        The waveform data for a multiplex group as an :class:`~numpy.ndarray`
        with shape (samples, channels).
    """
    if "WaveformSequence" not in ds:
        raise AttributeError(
            "No (5400,0100) Waveform Sequence element found in the dataset"
        )

    for ii, item in enumerate(cast(list["Dataset"], ds.WaveformSequence)):
        required_elements = [
            "NumberOfWaveformChannels",
            "NumberOfWaveformSamples",
            "WaveformBitsAllocated",
            "WaveformSampleInterpretation",
            "WaveformData",
        ]
        missing = [elem for elem in required_elements if elem not in item]
        if missing:
            raise AttributeError(
                f"Unable to convert the waveform multiplex group with index "
                f"{ii} as the following required elements are missing from "
                f"the sequence item: {', '.join(missing)}"
            )

        # Determine the expected length of the data (without padding)
        bytes_per_sample = cast(int, item.WaveformBitsAllocated) // 8
        nr_samples = cast(int, item.NumberOfWaveformSamples)
        nr_channels = cast(int, item.NumberOfWaveformChannels)
        bits_allocated = cast(int, item.WaveformBitsAllocated)
        sample_interpretation = cast(str, item.WaveformSampleInterpretation)
        expected_len = nr_samples * nr_channels * bytes_per_sample

        # Waveform Data is ordered as (C = channel, S = sample):
        # C1S1, C2S1, ..., CnS1, C1S2, ..., CnS2, ..., C1Sm, ..., CnSm
        dtype = WAVEFORM_DTYPES[(bits_allocated, sample_interpretation)]
        arr = np.frombuffer(cast(bytes, item.WaveformData)[:expected_len], dtype=dtype)
        # Reshape to (samples, channels) and make writeable
        arr = np.copy(arr.reshape(nr_samples, nr_channels))

        if not as_raw:
            # Apply correction factor (if possible)
            arr = arr.astype("<f8")
            seq = cast(list["Dataset"], item.ChannelDefinitionSequence)
            for jj, ch in enumerate(seq):
                baseline = ch.get("ChannelBaseline", 0.0)
                sensitivity = ch.get("ChannelSensitivity", 1.0)
                correction = ch.get("ChannelSensitivityCorrectionFactor", 1.0)
                arr[..., jj] = arr[..., jj] * sensitivity * correction + baseline

        yield arr


def multiplex_array(ds: "Dataset", index: int, as_raw: bool = True) -> "np.ndarray":
    """Return an :class:`~numpy.ndarray` for the multiplex group in the
    *Waveform Sequence* at `index`.

    .. versionadded:: 2.1

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The :class:`Dataset` containing a :dcm:`Waveform
        <part03/sect_C.10.9.html>` module and the *Waveform Sequence* to be
        converted.
    index : int
        The index of the multiplex group to return.
    as_raw : bool, optional
        If ``True`` (default), then return the raw unitless waveform data. If
        ``False`` then attempt to convert the raw data for each channel to the
        quantity specified by the corresponding (003A,0210) *Channel
        Sensitivity* unit.

    Returns
    -------
    np.ndarray
        The waveform data for a multiplex group as an :class:`~numpy.ndarray`
        with shape (samples, channels).
    """
    if "WaveformSequence" not in ds:
        raise AttributeError(
            "No (5400,0100) Waveform Sequence element found in the dataset"
        )

    item = cast(list["Dataset"], ds.WaveformSequence)[index]
    required_elements = [
        "NumberOfWaveformChannels",
        "NumberOfWaveformSamples",
        "WaveformBitsAllocated",
        "WaveformSampleInterpretation",
        "WaveformData",
    ]
    missing = [elem for elem in required_elements if elem not in item]
    if missing:
        raise AttributeError(
            f"Unable to convert the waveform multiplex group with index "
            f"{index} as the following required elements are missing from "
            f"the sequence item: {', '.join(missing)}"
        )

    # Determine the expected length of the data (without padding)
    bytes_per_sample = cast(int, item.WaveformBitsAllocated) // 8
    nr_samples = cast(int, item.NumberOfWaveformSamples)
    nr_channels = cast(int, item.NumberOfWaveformChannels)
    bits_allocated = cast(int, item.WaveformBitsAllocated)
    sample_interpretation = cast(str, item.WaveformSampleInterpretation)
    expected_len = nr_samples * nr_channels * bytes_per_sample

    # Waveform Data is ordered as (C = channel, S = sample):
    # C1S1, C2S1, ..., CnS1, C1S2, ..., CnS2, ..., C1Sm, ..., CnSm
    dtype = WAVEFORM_DTYPES[(bits_allocated, sample_interpretation)]
    arr = np.frombuffer(cast(bytes, item.WaveformData)[:expected_len], dtype=dtype)
    # Reshape to (samples, channels) and make writeable
    arr = np.copy(arr.reshape(nr_samples, nr_channels))

    if not as_raw:
        # Apply correction factor (if possible)
        arr = arr.astype("<f8")
        seq = cast(list["Dataset"], item.ChannelDefinitionSequence)
        for jj, ch in enumerate(seq):
            baseline = ch.get("ChannelBaseline", 0.0)
            sensitivity = ch.get("ChannelSensitivity", 1.0)
            correction = ch.get("ChannelSensitivityCorrectionFactor", 1.0)
            arr[..., jj] = arr[..., jj] * sensitivity * correction + baseline

    return cast("np.ndarray", arr)


def _waveform_data_bytes(
    item: "Dataset", fp: BinaryIO | None, first: int, last: int
) -> bytes:
    """Return the bytes of the (5400,1010) *Waveform Data* element for the
    sample range `[first, last)`.

    If `fp` is given and the element's value has been deferred, only the
    required byte range is read from the (seekable) file object.
    """
    nr_channels = cast(int, item.NumberOfWaveformChannels)
    bytes_per_sample = cast(int, item.WaveformBitsAllocated) // 8
    frame_len = nr_channels * bytes_per_sample
    byte_offset = first * frame_len
    byte_len = (last - first) * frame_len

    if fp is not None:
        from pydicom.dataelem import RawDataElement

        raw = item.get_item(0x54001010, keep_deferred=True)
        if isinstance(raw, RawDataElement) and raw.value is None:
            # Deferred element: seek to the window and read only what's needed
            fp.seek(raw.value_tell + byte_offset)
            buf = fp.read(byte_len)
            if len(buf) != byte_len:
                raise ValueError(
                    "Unable to read the waveform data window from the file "
                    f"source, expected {byte_len} bytes, got {len(buf)}"
                )

            return buf

    data = cast(bytes, item.WaveformData)
    return data[byte_offset : byte_offset + byte_len]


def waveform_window(
    src: "Dataset | str | os.PathLike[str] | BinaryIO",
    start: float,
    end: float,
    channels: Sequence[int] | None = None,
    group: int = 0,
    as_raw: bool = False,
) -> WaveformWindow:
    """Read a time window of selected channels from a multiplex group in the
    (5400,0100) *Waveform Sequence*.

    .. versionadded:: 3.1

    The window is the half-open interval `[start, end)` in seconds: a sample
    is included if its acquisition time `t` fulfils `start <= t < end`.
    Sample times are given by the (0018,1068) *Multiplex Group Time Offset*
    (if present), the (003A,001A) *Sampling Frequency* and the per-channel
    (003A,0214) *Channel Time Skew* and (003A,0215) *Channel Sample Skew*.

    Parameters
    ----------
    src : pydicom.dataset.Dataset, str, PathLike or file-like
        The source to read from. May be a :class:`~pydicom.dataset.Dataset`
        containing a :dcm:`Waveform <part03/sect_C.10.9.html>` module, a
        path to a DICOM file, or a seekable file-like object positioned at
        the start of a DICOM file. For paths and file-like objects the
        dataset is read lazily and only the byte range of the (5400,1010)
        *Waveform Data* element covering the requested window is loaded.
    start : float
        The start of the time window in seconds (inclusive).
    end : float
        The end of the time window in seconds (exclusive).
    channels : sequence of int, optional
        The indices of the channels to return. If `None` (default), return
        all channels of the multiplex group.
    group : int, optional
        The index of the multiplex group in the *Waveform Sequence* to read
        (default `0`).
    as_raw : bool, optional
        If `True`, return the raw unitless samples. If `False` (default),
        calibrate each channel using its (003A,0210) *Channel Sensitivity*,
        (003A,0212) *Channel Sensitivity Correction Factor* and (003A,0213)
        *Channel Baseline* as `raw * sensitivity * correction + baseline`.

    Returns
    -------
    WaveformWindow
        A named tuple with the `times` of the returned samples in seconds
        and the `samples` themselves, with shape `(samples, channels)`.
        If all selected channels share the same skew the time axis has shape
        `(samples,)`, otherwise `(samples, channels)`. A window that
        doesn't overlap the multiplex group returns empty arrays.
    """
    if not HAVE_NP:
        raise RuntimeError("The waveform data handler requires numpy")

    if not end > start:
        raise ValueError(f"'end' ({end}) must be greater than 'start' ({start})")

    from pydicom.dataelem import RawDataElement
    from pydicom.dataset import Dataset

    fp: BinaryIO | None = None
    close_fp = False
    file_ds: FileDataset | None = None
    if isinstance(src, Dataset):
        ds = src
    else:
        from pydicom.filereader import dcmread

        if isinstance(src, str | os.PathLike):
            fp = open(src, "rb")
            close_fp = True
        else:
            fp = cast(BinaryIO, src)

        # Defer reading large element values so only the required section
        #   of the Waveform Data is loaded from the file
        file_ds = dcmread(fp, defer_size=1024)
        ds = file_ds

    try:
        if file_ds is not None:
            from pydicom.charset import default_encoding
            from pydicom.filereader import read_sequence

            sq_elem = file_ds.get_item(0x54000100, keep_deferred=True)
            if sq_elem is None:
                raise AttributeError(
                    "No (5400,0100) Waveform Sequence element found in the dataset"
                )

            if isinstance(sq_elem, RawDataElement) and sq_elem.value is None:
                # Defined length sequence with deferred value: parse the
                #   items from the file directly, keeping the deferred
                #   Waveform Data elements' file offsets
                is_implicit, is_little = cast(
                    "tuple[bool, bool]", file_ds.original_encoding
                )
                cast(BinaryIO, fp).seek(sq_elem.value_tell)
                groups = read_sequence(
                    fp,
                    is_implicit,
                    is_little,
                    sq_elem.length,
                    default_encoding,
                    defer_size=1024,
                )
            else:
                # Undefined length sequence: already parsed while reading,
                #   with the items' Waveform Data elements deferred
                groups = cast(list["Dataset"], file_ds.WaveformSequence)

            # Allow deferred conversion of any deferred elements in the
            #   (plain) item datasets
            from pydicom.filereader import _set_sequence_item_file_context

            for group_item in groups:
                group_item.filename = file_ds.filename  # type: ignore[attr-defined]
                group_item.buffer = file_ds.buffer  # type: ignore[attr-defined]
                group_item.fileobj_type = file_ds.fileobj_type  # type: ignore[attr-defined]
                group_item.timestamp = file_ds.timestamp  # type: ignore[attr-defined]
                _set_sequence_item_file_context(group_item)  # type: ignore[arg-type]
        else:
            if "WaveformSequence" not in ds:
                raise AttributeError(
                    "No (5400,0100) Waveform Sequence element found in the dataset"
                )

            groups = cast(list["Dataset"], ds.WaveformSequence)

        item = groups[group]
        required_elements = [
            "NumberOfWaveformChannels",
            "NumberOfWaveformSamples",
            "WaveformBitsAllocated",
            "WaveformSampleInterpretation",
            "WaveformData",
            "SamplingFrequency",
        ]
        missing = [elem for elem in required_elements if elem not in item]
        if missing:
            raise AttributeError(
                f"Unable to read the waveform multiplex group with index "
                f"{group} as the following required elements are missing "
                f"from the sequence item: {', '.join(missing)}"
            )

        nr_channels = cast(int, item.NumberOfWaveformChannels)
        nr_samples = cast(int, item.NumberOfWaveformSamples)
        frequency = float(cast(Any, item.SamplingFrequency))
        bits_allocated = cast(int, item.WaveformBitsAllocated)
        interpretation = cast(str, item.WaveformSampleInterpretation)
        try:
            dtype = np.dtype(WAVEFORM_DTYPES[(bits_allocated, interpretation)])
        except KeyError:
            raise ValueError(
                f"Unsupported (5400,1004) Waveform Bits Allocated "
                f"'{bits_allocated}' and (5400,1006) Waveform Sample "
                f"Interpretation '{interpretation}' combination"
            ) from None

        if channels is None:
            channels = list(range(nr_channels))
        else:
            channels = list(channels)
            out_of_range = [ii for ii in channels if not 0 <= ii < nr_channels]
            if out_of_range:
                raise IndexError(
                    f"Channel index/indices {out_of_range} out of range for "
                    f"a multiplex group with {nr_channels} channels"
                )

        # Sample times are origin + i / frequency, the window is [start, end)
        offset_ms = float(cast(Any, item.get("MultiplexGroupTimeOffset", 0.0)))
        origin = offset_ms / 1000.0
        first, last = _window_indices(start, end, origin, frequency, nr_samples)

        # Per-channel timing and calibration parameters
        definitions = cast(list["Dataset"], item.get("ChannelDefinitionSequence", []))
        skews = []
        gains = []
        baselines = []
        for ii in channels:
            ch = definitions[ii] if ii < len(definitions) else None
            time_skew = float(ch.get("ChannelTimeSkew", 0.0)) if ch else 0.0
            sample_skew = float(ch.get("ChannelSampleSkew", 0.0)) if ch else 0.0
            skews.append(time_skew + sample_skew / frequency)
            sensitivity = float(ch.get("ChannelSensitivity", 1.0)) if ch else 1.0
            correction = (
                float(ch.get("ChannelSensitivityCorrectionFactor", 1.0)) if ch else 1.0
            )
            gains.append(sensitivity * correction)
            baselines.append(float(ch.get("ChannelBaseline", 0.0)) if ch else 0.0)

        # Read only the samples within the window; any trailing padding of
        #   the Waveform Data (such as a Waveform Padding Value byte) lies
        #   beyond nr_samples and is never included
        buf = _waveform_data_bytes(item, fp, first, last)
        arr = np.frombuffer(buf, dtype=dtype).reshape(last - first, nr_channels)
        arr = np.copy(arr[:, channels])

        times = origin + (first + np.arange(last - first)) / frequency
        if skews and all(skew == skews[0] for skew in skews):
            times = times + skews[0]
        elif skews:
            times = times[:, None] + np.asarray(skews)

        if not as_raw:
            arr = arr.astype("<f8") * np.asarray(gains) + np.asarray(baselines)

        return WaveformWindow(times=times, samples=arr)
    finally:
        if close_fp:
            cast(BinaryIO, fp).close()
