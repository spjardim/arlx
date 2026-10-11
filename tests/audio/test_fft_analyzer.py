import numpy as np
import pytest

from arlx.audio.fft_analyzer import (
    DEFAULT_FREQUENCY_BANDS,
    FLOOR_DB,
    ATTACK_DB_PER_BLOCK,
    DECAY_DB_PER_BLOCK,
    FFTAnalyzer,
    FrequencyBand,
    FrequencyBands,
)

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024


def sine_block(frequency: float, amplitude: float = 1.0, n=BLOCK_SIZE, sample_rate=SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def silence_block(n=BLOCK_SIZE) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def bin_frequency_in_band(band, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE) -> float:
    spec_x = np.fft.fftfreq(block_size, d=1.0 / sample_rate)[:block_size // 2]
    in_band = spec_x[(spec_x >= band.low_freq) & (spec_x < band.high_freq)]
    assert len(in_band) > 0, f"no FFT bin falls inside band '{band.name}'"
    return float(in_band[len(in_band) // 2])


# ---- FrequencyBand ----

def test_frequency_band_rejects_low_greater_than_high():
    with pytest.raises(ValueError):
        FrequencyBand("Bad", 100, 50)


def test_frequency_band_rejects_low_equal_to_high():
    with pytest.raises(ValueError):
        FrequencyBand("Bad", 100, 100)


def test_frequency_band_lt_and_gt_by_disjoint_ranges():
    low = FrequencyBand("Low", 0, 100)
    high = FrequencyBand("High", 100, 200)
    assert low < high
    assert high > low
    assert not (high < low)
    assert not (low > high)


def test_frequency_band_eq_by_bounds():
    assert FrequencyBand("A", 0, 100) == FrequencyBand("B", 0, 100)
    assert FrequencyBand("A", 0, 100) != FrequencyBand("B", 0, 200)


def test_frequency_band_comparisons_return_not_implemented_for_other_types():
    band = FrequencyBand("A", 0, 100)
    assert band.__lt__(5) is NotImplemented
    assert band.__gt__(5) is NotImplemented
    assert band.__eq__(5) is NotImplemented


def test_frequency_band_repr_contains_fields():
    band = FrequencyBand("Bass", 60, 250)
    assert repr(band) == "FrequencyBand(name='Bass', low_freq=60, high_freq=250)"


# ---- FrequencyBands ----

def test_frequency_bands_sorts_by_low_freq():
    bands = FrequencyBands([
        FrequencyBand("High", 200, 300),
        FrequencyBand("Low", 0, 100),
    ])
    assert [band.name for band in bands.bands] == ["Low", "High"]


def test_frequency_bands_rejects_overlapping_ranges():
    with pytest.raises(ValueError):
        FrequencyBands([
            FrequencyBand("A", 0, 150),
            FrequencyBand("B", 100, 200),
        ])


def test_frequency_bands_allows_adjacent_touching_ranges():
    bands = FrequencyBands([
        FrequencyBand("A", 0, 100),
        FrequencyBand("B", 100, 200),
    ])
    assert len(bands.bands) == 2


def test_default_frequency_bands_has_seven_non_overlapping_bands():
    assert len(DEFAULT_FREQUENCY_BANDS.bands) == 7


# ---- FFTAnalyzer: accessor copy semantics ----

@pytest.mark.parametrize("getter_name", [
    "get_raw_fft_data",
    "get_mean_fft_bins",
    "get_rms_fft_bins",
    "get_log_rms_fft_bins",
    "get_norm_fft_bins",
    "get_smoothed_norm_bins",
])
def test_getters_return_copies_not_internal_arrays(getter_name):
    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    analyzer.process(sine_block(1000))

    result = getattr(analyzer, getter_name)()
    result[:] = -999

    result_again = getattr(analyzer, getter_name)()
    assert not np.any(result_again == -999)


# ---- FFTAnalyzer: band energy assignment ----

@pytest.mark.parametrize("band_index", range(7))
def test_band_energy_peaks_in_the_matching_band(band_index):
    band = DEFAULT_FREQUENCY_BANDS.bands[band_index]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    analyzer.process(sine_block(frequency))

    assert np.argmax(analyzer.get_mean_fft_bins()) == band_index
    assert np.argmax(analyzer.get_rms_fft_bins()) == band_index


def test_log_rms_matches_manual_calculation_from_raw_fft_data():
    band = DEFAULT_FREQUENCY_BANDS.bands[3]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    analyzer.process(sine_block(frequency))

    spec_x = np.fft.fftfreq(BLOCK_SIZE, d=1.0 / SAMPLE_RATE)[:BLOCK_SIZE // 2]
    band_indices = np.where((spec_x >= band.low_freq) & (spec_x < band.high_freq))[0]
    expected_rms = np.sqrt(np.mean(analyzer.get_raw_fft_data()[band_indices] ** 2))
    expected_log_rms = 20 * np.log10(max(expected_rms, 1e-6))

    assert analyzer.get_log_rms_fft_bins()[3] == pytest.approx(expected_log_rms)


# ---- FFTAnalyzer: running-max attack/decay envelope ----

def test_running_max_db_attacks_by_exactly_one_step_on_first_loud_block():
    band = DEFAULT_FREQUENCY_BANDS.bands[0]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    analyzer.process(sine_block(frequency, amplitude=1.0))

    assert analyzer.running_max_db[0] == pytest.approx(FLOOR_DB + ATTACK_DB_PER_BLOCK)


def test_running_max_db_rises_monotonically_while_signal_stays_loud():
    band = DEFAULT_FREQUENCY_BANDS.bands[0]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    previous = FLOOR_DB
    for _ in range(5):
        analyzer.process(sine_block(frequency, amplitude=1.0))
        current = analyzer.running_max_db[0]
        assert current > previous
        previous = current


def test_running_max_db_decays_by_exactly_one_step_on_silence_after_being_loud():
    band = DEFAULT_FREQUENCY_BANDS.bands[0]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    analyzer.process(sine_block(frequency, amplitude=1.0))
    loud_running_max = analyzer.running_max_db[0]

    analyzer.process(silence_block())
    assert analyzer.running_max_db[0] == pytest.approx(loud_running_max - DECAY_DB_PER_BLOCK)


def test_running_max_db_floors_out_at_floor_db():
    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    for _ in range(1000):
        analyzer.process(silence_block())
    assert analyzer.running_max_db[0] == pytest.approx(FLOOR_DB)


# ---- FFTAnalyzer: normalization ----

def test_norm_fft_bins_clips_to_one_when_log_rms_far_exceeds_running_max():
    band = DEFAULT_FREQUENCY_BANDS.bands[0]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    analyzer.process(sine_block(frequency, amplitude=1.0))

    assert analyzer.get_norm_fft_bins()[0] == pytest.approx(1.0)


def test_norm_fft_bins_is_zero_while_running_max_still_at_floor():
    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    analyzer.process(silence_block())
    assert analyzer.get_norm_fft_bins()[0] == 0.0


def test_norm_fft_bins_stay_within_zero_and_one_bounds():
    band = DEFAULT_FREQUENCY_BANDS.bands[0]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    for _ in range(20):
        analyzer.process(sine_block(frequency, amplitude=np.random.default_rng(0).uniform(0, 1)))
        norm = analyzer.get_norm_fft_bins()
        assert np.all(norm >= 0.0) and np.all(norm <= 1.0)


# ---- FFTAnalyzer: temporal smoothing ----

def test_smoothed_norm_bins_moves_by_attack_fraction_toward_norm():
    band = DEFAULT_FREQUENCY_BANDS.bands[0]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, smooth_attack=0.5)
    analyzer.process(sine_block(frequency, amplitude=1.0))

    norm = analyzer.get_norm_fft_bins()[0]
    assert analyzer.get_smoothed_norm_bins()[0] == pytest.approx(0.5 * norm)


def test_smoothed_norm_bins_moves_by_decay_fraction_toward_norm():
    band = DEFAULT_FREQUENCY_BANDS.bands[0]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, smooth_decay=0.1)
    analyzer.process(sine_block(frequency, amplitude=1.0))
    smoothed_while_loud = analyzer.get_smoothed_norm_bins()[0]

    analyzer.process(silence_block())
    expected = smoothed_while_loud + 0.1 * (analyzer.get_norm_fft_bins()[0] - smoothed_while_loud)
    assert analyzer.get_smoothed_norm_bins()[0] == pytest.approx(expected)


# ---- FFTAnalyzer: band with no FFT bins in range ----

def test_band_with_no_fft_bins_in_range_reports_floor_and_zero():
    empty_band = FrequencyBands([FrequencyBand("TooNarrow", 10, 20)])
    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, frequency_bands=empty_band)

    analyzer.process(sine_block(1000, amplitude=1.0))

    assert analyzer.get_mean_fft_bins()[0] == 0
    assert analyzer.get_rms_fft_bins()[0] == 0
    assert analyzer.get_log_rms_fft_bins()[0] == FLOOR_DB
    assert analyzer.get_norm_fft_bins()[0] == 0
    assert analyzer.running_max_db[0] == pytest.approx(FLOOR_DB)


# ---- FFTAnalyzer: known limitation, documented rather than fixed here ----

def test_more_than_seven_bands_raises_index_error():
    # band bins are hardcoded to length 7 in __init__ regardless of how many
    # bands are actually configured; this pins the current (buggy) behavior
    # so a future fix is a deliberate choice rather than a silent change.
    many_bands = FrequencyBands([
        FrequencyBand(f"Band{i}", i * 1000, (i + 1) * 1000) for i in range(8)
    ])
    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, frequency_bands=many_bands)

    with pytest.raises(IndexError):
        analyzer.process(sine_block(1000))
