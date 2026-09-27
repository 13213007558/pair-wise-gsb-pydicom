# Copyright 2008-2020 pydicom authors. See LICENSE file for details.


from io import BytesIO

import pytest

from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.filereader import dcmread
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

try:
    import numpy as np

    HAVE_NP = True
except ImportError:
    HAVE_NP = False

try:
    from pydicom.waveforms import numpy_handler as NP_HANDLER
    from pydicom.waveforms.numpy_handler import (
        WaveformWindow,
        generate_multiplex,
        multiplex_array,
        waveform_window,
    )
except ImportError:
    NP_HANDLER = None


ECG = get_testdata_file("waveform_ecg.dcm")


@pytest.mark.skipif(HAVE_NP, reason="Numpy is available")
def test_waveform_array_raises():
    """Test waveform_array raises exception for all syntaxes."""
    ds = dcmread(ECG)
    msg = r"The waveform data handler requires numpy"
    with pytest.raises(RuntimeError, match=msg):
        ds.waveform_array(0)


@pytest.mark.skipif(not HAVE_NP, reason="Numpy not available")
def test_simple():
    """Simple functionality test."""
    ds = dcmread(ECG)
    ds.waveform_array(index=0)
    ds.waveform_array(index=1)


@pytest.mark.skipif(not HAVE_NP, reason="Numpy not available")
class TestHandlerGenerateMultiplex:
    """Tests for the waveform numpy_handler.generate_multiplex."""

    def test_no_waveform_sequence(self):
        """Test that missing waveform sequence raises exception."""
        ds = dcmread(ECG)
        del ds.WaveformSequence
        msg = r"No \(5400,0100\) Waveform Sequence element found in the dataset"
        gen = generate_multiplex(ds)
        with pytest.raises(AttributeError, match=msg):
            next(gen)

    def test_missing_required(self):
        """Test that missing required element in sequence raises exception."""
        ds = dcmread(ECG)
        item = ds.WaveformSequence[0]
        del item.NumberOfWaveformSamples
        msg = (
            "Unable to convert the waveform multiplex group with index "
            "0 as the following required elements are missing from "
            "the sequence item: NumberOfWaveformSamples"
        )
        gen = generate_multiplex(ds)
        with pytest.raises(AttributeError, match=msg):
            next(gen)

    def test_as_raw(self):
        """Test that as_raw=True works as expected."""
        ds = dcmread(ECG)
        gen = generate_multiplex(ds, as_raw=True)
        arr = next(gen)
        assert [80, 65, 50, 35, 37] == arr[0:5, 0].tolist()
        assert [90, 85, 80, 75, 77] == arr[0:5, 1].tolist()
        assert arr.dtype == "<i2"
        assert arr.flags.writeable
        assert (10000, 12) == arr.shape

    def test_not_as_raw(self):
        """Test that as_raw=False works as expected."""
        ds = dcmread(ECG)
        gen = generate_multiplex(ds, as_raw=False)
        arr = next(gen)
        assert [100, 81.25, 62.5, 43.75, 46.25] == arr[0:5, 0].tolist()
        assert [112.5, 106.25, 100, 93.75, 96.25] == arr[0:5, 1].tolist()
        assert arr.dtype == "<f8"
        assert arr.flags.writeable
        assert (10000, 12) == arr.shape


def _multiplex_item():
    """Return a multiplex group item with 3 channels, 500 Hz, 10000 samples.

    The group starts 1 s after the nominal start time and each channel has
    a different gain, offset and skew:

    * Channel 0: gain 1.0, offset 0.0, no skew
    * Channel 1: gain 1.0 (0.5 * 2.0), offset -100.0, time skew 2 ms
    * Channel 2: gain 2.0 (2.5 * 0.8), offset 10.0, sample skew 1 (2 ms)
    """
    item = Dataset()
    item.NumberOfWaveformChannels = 3
    item.NumberOfWaveformSamples = 10000
    item.SamplingFrequency = "500"
    item.MultiplexGroupTimeOffset = "1000"  # ms
    item.WaveformBitsAllocated = 16
    item.WaveformSampleInterpretation = "SS"

    definitions = []
    ch = Dataset()
    ch.ChannelSensitivity = "1.0"
    ch.ChannelSensitivityCorrectionFactor = "1.0"
    ch.ChannelBaseline = "0.0"
    definitions.append(ch)
    ch = Dataset()
    ch.ChannelSensitivity = "0.5"
    ch.ChannelSensitivityCorrectionFactor = "2.0"
    ch.ChannelBaseline = "-100.0"
    ch.ChannelTimeSkew = "0.002"
    definitions.append(ch)
    ch = Dataset()
    ch.ChannelSensitivity = "2.5"
    ch.ChannelSensitivityCorrectionFactor = "0.8"
    ch.ChannelBaseline = "10.0"
    ch.ChannelSampleSkew = "1"
    definitions.append(ch)
    item.ChannelDefinitionSequence = definitions

    nr_samples = 10000
    ch0 = np.arange(nr_samples, dtype="<i2") % 1000
    ch1 = -(np.arange(nr_samples, dtype="<i2") % 500)
    ch2 = np.full(nr_samples, 1000, dtype="<i2")
    ch2[-1] = 0  # valid zero as the last sample
    item.WaveformData = np.column_stack([ch0, ch1, ch2]).tobytes()

    return item


def _waveform_ds():
    ds = Dataset()
    ds.WaveformSequence = [_multiplex_item()]
    return ds


def _waveform_file_ds():
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.WaveformSequence = [_multiplex_item()]
    return ds


class _ReadSpy:
    """File-like wrapper that records the size of every read."""

    def __init__(self, fp):
        self._fp = fp
        self.reads = []

    def read(self, size=-1):
        data = self._fp.read(size)
        self.reads.append(len(data))
        return data

    def seek(self, offset, whence=0):
        return self._fp.seek(offset, whence)

    def tell(self):
        return self._fp.tell()


@pytest.mark.skipif(not HAVE_NP, reason="Numpy not available")
class TestWaveformWindow:
    """Tests for the waveform numpy_handler.waveform_window."""

    def test_raw_window(self):
        """Test raw samples and time axis for a channel subset."""
        result = waveform_window(
            _waveform_ds(), 10.0, 12.0, channels=[0, 2], as_raw=True
        )
        assert isinstance(result, WaveformWindow)
        assert (1000, 2) == result.samples.shape
        assert result.samples.dtype == "<i2"
        assert result.samples.flags.writeable
        # Sample 4500 of channel 0 is 4500 % 1000
        assert 500 == result.samples[0, 0]
        assert 499 == result.samples[-1, 0]
        assert 1000 == result.samples[0, 1]
        # Channels 0 and 2 have different skews -> per-channel time axis
        assert (1000, 2) == result.times.shape
        assert 10.0 == pytest.approx(result.times[0, 0])
        assert 10.002 == pytest.approx(result.times[0, 1])
        assert 11.998 == pytest.approx(result.times[-1, 0])
        assert 12.0 == pytest.approx(result.times[-1, 1])

    def test_calibrated_window(self):
        """Test calibrated values against the raw data."""
        ds = _waveform_ds()
        raw = waveform_window(ds, 10.0, 12.0, as_raw=True).samples
        result = waveform_window(ds, 10.0, 12.0)
        assert result.samples.dtype == "<f8"
        # Channel 0: raw * 1.0 * 1.0 + 0.0
        np.testing.assert_allclose(result.samples[:, 0], raw[:, 0])
        # Channel 1: raw * 0.5 * 2.0 - 100.0
        np.testing.assert_allclose(result.samples[:, 1], raw[:, 1] - 100.0)
        # Channel 2: raw * 2.5 * 0.8 + 10.0
        np.testing.assert_allclose(result.samples[:, 2], raw[:, 2] * 2.0 + 10.0)

    def test_shared_skew_1d_times(self):
        """Test a 1D time axis is returned when all skews are equal."""
        result = waveform_window(
            _waveform_ds(), 10.0, 12.0, channels=[1, 2], as_raw=True
        )
        assert (1000,) == result.times.shape
        assert 10.002 == pytest.approx(result.times[0])
        assert 12.0 == pytest.approx(result.times[-1])

    def test_group_time_offset(self):
        """Test the multiplex group time offset is applied."""
        result = waveform_window(_waveform_ds(), 1.0, 1.01, channels=[0])
        assert 1.0 == pytest.approx(result.times[0])
        assert 1.008 == pytest.approx(result.times[-1])

    def test_half_open_interval(self):
        """Test boundary samples with float-computed window endpoints."""
        ds = _waveform_ds()
        # 1.0 + 4500 * (1 / 500) and 1.0 + 5500 * (1 / 500) may not be
        #   exactly 10.0 and 12.0 due to floating point rounding
        start = 1.0 + 4500 * (1.0 / 500.0)
        end = 1.0 + 5500 * (1.0 / 500.0)
        result = waveform_window(ds, start, end, channels=[0], as_raw=True)
        assert (1000, 1) == result.samples.shape
        # The sample at 'start' is included, the one at 'end' excluded
        assert 500 == result.samples[0, 0]  # sample 4500
        assert 499 == result.samples[-1, 0]  # sample 5499

    def test_single_sample_window(self):
        """Test a window covering exactly one sample."""
        start = 1.0 + 4500 / 500
        result = waveform_window(
            _waveform_ds(), start, start + 0.002, channels=[0], as_raw=True
        )
        assert (1, 1) == result.samples.shape
        assert 500 == result.samples[0, 0]

    def test_window_outside_range_empty(self):
        """Test windows not overlapping the data return empty arrays."""
        ds = _waveform_ds()
        result = waveform_window(ds, 100.0, 101.0, channels=[0])
        assert (0, 1) == result.samples.shape
        assert (0,) == result.times.shape
        result = waveform_window(ds, -5.0, -1.0, channels=[0])
        assert (0, 1) == result.samples.shape

    def test_padding_distinguished_from_zero(self):
        """Test waveform padding is not returned as a valid zero sample."""
        item = Dataset()
        item.NumberOfWaveformChannels = 1
        item.NumberOfWaveformSamples = 5
        item.SamplingFrequency = "1000"
        item.WaveformBitsAllocated = 8
        item.WaveformSampleInterpretation = "UB"
        item.ChannelDefinitionSequence = [Dataset()]
        # Odd number of data bytes, padded with a zero padding byte; the
        #   third sample is a valid zero
        item.WaveformData = bytes([10, 20, 0, 30, 40]) + b"\x00"
        item.WaveformPaddingValue = b"\x00"
        ds = Dataset()
        ds.WaveformSequence = [item]

        result = waveform_window(ds, 0.0, 1.0, as_raw=True)
        assert (5, 1) == result.samples.shape
        assert [10, 20, 0, 30, 40] == result.samples[:, 0].tolist()

    def test_sample_interpretations(self):
        """Test different sample interpretations are decoded correctly."""
        for bits, interpretation, dtype in [
            (8, "UB", "u1"),
            (16, "US", "<u2"),
            (32, "SL", "<i4"),
        ]:
            item = Dataset()
            item.NumberOfWaveformChannels = 2
            item.NumberOfWaveformSamples = 100
            item.SamplingFrequency = "100"
            item.WaveformBitsAllocated = bits
            item.WaveformSampleInterpretation = interpretation
            item.ChannelDefinitionSequence = [Dataset(), Dataset()]
            arr = np.arange(200, dtype=dtype).reshape(100, 2)
            item.WaveformData = arr.tobytes()
            ds = Dataset()
            ds.WaveformSequence = [item]

            result = waveform_window(ds, 0.1, 0.5, as_raw=True)
            assert (40, 2) == result.samples.shape
            assert result.samples.dtype == np.dtype(dtype)
            assert (result.samples == arr[10:50]).all()

    def test_file_like_partial_read(self):
        """Test only the required Waveform Data section is read."""
        bio = BytesIO()
        _waveform_file_ds().save_as(bio, enforce_file_format=True)
        bio.seek(0)
        spy = _ReadSpy(bio)

        result = waveform_window(spy, 10.0, 12.0, channels=[0, 2], as_raw=True)
        assert (1000, 2) == result.samples.shape
        assert 500 == result.samples[0, 0]
        assert 1000 == result.samples[0, 1]
        assert 10.0 == pytest.approx(result.times[0, 0])

        window_len = 1000 * 3 * 2  # samples * channels * bytes per sample
        full_len = 10000 * 3 * 2
        # The window is read in a single read and the full Waveform Data
        #   element is never loaded
        assert window_len in spy.reads
        assert all(size < full_len for size in spy.reads)

    def test_file_like_matches_dataset(self):
        """Test file-like and dataset sources return the same data."""
        bio = BytesIO()
        _waveform_file_ds().save_as(bio, enforce_file_format=True)
        expected = waveform_window(_waveform_ds(), 10.0, 12.0)
        bio.seek(0)
        result = waveform_window(bio, 10.0, 12.0)
        assert (result.samples == expected.samples).all()
        assert (result.times == expected.times).all()

    def test_path_source(self, tmp_path):
        """Test reading from a path."""
        path = tmp_path / "waveform.dcm"
        _waveform_file_ds().save_as(path, enforce_file_format=True)
        result = waveform_window(path, 10.0, 12.0, channels=[1], as_raw=True)
        assert (1000, 1) == result.samples.shape
        assert 0 == result.samples[0, 0]  # -(4500 % 500)
        assert -499 == result.samples[-1, 0]  # -(5499 % 500)
        assert 10.002 == pytest.approx(result.times[0])

    def test_testdata_file(self):
        """Test reading a window from the ECG test data file."""
        full = multiplex_array(dcmread(ECG), 0, as_raw=True)
        result = waveform_window(ECG, 1.0, 2.0, channels=[0, 3], as_raw=True)
        assert (1000, 2) == result.samples.shape
        assert 1.0 == pytest.approx(result.times[0])
        assert 1.999 == pytest.approx(result.times[-1])
        assert (result.samples == full[1000:2000][:, [0, 3]]).all()

    def test_end_not_after_start_raises(self):
        """Test end <= start raises an exception."""
        ds = _waveform_ds()
        with pytest.raises(ValueError, match="must be greater than 'start'"):
            waveform_window(ds, 12.0, 10.0)
        with pytest.raises(ValueError, match="must be greater than 'start'"):
            waveform_window(ds, 10.0, 10.0)

    def test_channel_out_of_range_raises(self):
        """Test an invalid channel index raises an exception."""
        msg = "out of range for a multiplex group with 3 channels"
        with pytest.raises(IndexError, match=msg):
            waveform_window(_waveform_ds(), 10.0, 12.0, channels=[0, 3])

    def test_no_waveform_sequence_raises(self):
        """Test that a missing waveform sequence raises an exception."""
        msg = r"No \(5400,0100\) Waveform Sequence element found in the dataset"
        with pytest.raises(AttributeError, match=msg):
            waveform_window(Dataset(), 10.0, 12.0)

    def test_missing_required_raises(self):
        """Test that a missing required element raises an exception."""
        ds = _waveform_ds()
        del ds.WaveformSequence[0].SamplingFrequency
        msg = (
            "Unable to read the waveform multiplex group with index "
            "0 as the following required elements are missing "
            "from the sequence item: SamplingFrequency"
        )
        with pytest.raises(AttributeError, match=msg):
            waveform_window(ds, 10.0, 12.0)


@pytest.mark.skipif(not HAVE_NP, reason="Numpy not available")
class TestHandlerMultiplexArray:
    """Tests for the waveform numpy_handler.multiplex_array."""

    def test_no_waveform_sequence(self):
        """Test that missing waveform sequence raises exception."""
        ds = dcmread(ECG)
        del ds.WaveformSequence
        msg = r"No \(5400,0100\) Waveform Sequence element found in the dataset"
        with pytest.raises(AttributeError, match=msg):
            multiplex_array(ds, 0)

    def test_missing_required(self):
        """Test that missing required element in sequence raises exception."""
        ds = dcmread(ECG)
        item = ds.WaveformSequence[0]
        del item.NumberOfWaveformSamples
        msg = (
            "Unable to convert the waveform multiplex group with index "
            "0 as the following required elements are missing from "
            "the sequence item: NumberOfWaveformSamples"
        )
        with pytest.raises(AttributeError, match=msg):
            multiplex_array(ds, 0)

    def test_as_raw(self):
        """Test that as_raw=True works as expected."""
        ds = dcmread(ECG)
        arr = multiplex_array(ds, index=0, as_raw=True)
        assert [80, 65, 50, 35, 37] == arr[0:5, 0].tolist()
        assert [90, 85, 80, 75, 77] == arr[0:5, 1].tolist()
        assert arr.dtype == "<i2"
        assert arr.flags.writeable
        assert (10000, 12) == arr.shape

        arr = multiplex_array(ds, index=1, as_raw=True)
        assert [10, 10, 30, 35, 25] == arr[0:5, 0].tolist()
        assert [80, 80, 80, 85, 80] == arr[0:5, 1].tolist()
        assert arr.dtype == "<i2"
        assert arr.flags.writeable
        assert (1200, 12) == arr.shape

    def test_not_as_raw(self):
        """Test that as_raw=False works as expected."""
        ds = dcmread(ECG)
        arr = multiplex_array(ds, index=0, as_raw=False)
        assert [100, 81.25, 62.5, 43.75, 46.25] == arr[0:5, 0].tolist()
        assert [112.5, 106.25, 100, 93.75, 96.25] == arr[0:5, 1].tolist()
        assert arr.dtype == "<f8"
        assert arr.flags.writeable
        assert (10000, 12) == arr.shape
