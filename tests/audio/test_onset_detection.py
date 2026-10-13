import numpy as np
import pytest

from arlx.audio.fft_analyzer import DEFAULT_FREQUENCY_BANDS, FFTAnalyzer
from arlx.audio.onest_detection import OnsetDetection
from arlx.audio.spectral_flux import SpectralFlux

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
BLOCK_DURATION = BLOCK_SIZE / SAMPLE_RATE
NUM_BANDS = len(DEFAULT_FREQUENCY_BANDS.bands)


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


def make_detector(**overrides):
    params = dict(
        sample_rate=SAMPLE_RATE,
        hop_length=BLOCK_SIZE,
        window_duration=0.2,
        threshold_multiplier=2.0,
        refractory_duration=0.05,
    )
    params.update(overrides)
    return OnsetDetection(**params)


def warm_up(detector, values, num_bands=NUM_BANDS):
    for value in values:
        detector.process(value, np.full(num_bands, value))


# ---- construction ----

def test_history_length_derived_from_window_duration():
    detector = make_detector(window_duration=10 * BLOCK_DURATION)
    assert detector.history_length == 10


def test_history_length_has_minimum_of_one_block():
    detector = make_detector(window_duration=0.0)
    assert detector.history_length == 1


def test_refractory_blocks_derived_from_refractory_duration():
    detector = make_detector(refractory_duration=3 * BLOCK_DURATION)
    assert detector.refractory_blocks == 3


def test_refractory_blocks_has_minimum_of_one_block():
    detector = make_detector(refractory_duration=0.0)
    assert detector.refractory_blocks == 1


# ---- initial state / accessors ----

def test_initial_state_has_no_onsets_and_zero_thresholds():
    detector = make_detector()
    assert detector.get_global_onset() is False
    assert not detector.get_band_onsets().any()
    assert detector.get_global_threshold() == 0.0
    assert not detector.get_band_thresholds().any()


def test_get_band_onsets_returns_a_copy():
    detector = make_detector()
    detector.process(0.0, np.zeros(NUM_BANDS))
    onsets = detector.get_band_onsets()
    onsets[:] = True
    assert not detector.band_onsets.any()


def test_get_band_thresholds_returns_a_copy():
    detector = make_detector()
    thresholds = detector.get_band_thresholds()
    thresholds[:] = 999.0
    assert not detector.band_thresholds.any()


# ---- warm-up behavior ----

def test_global_onset_does_not_fire_during_warmup():
    detector = make_detector(window_duration=10 * BLOCK_DURATION)
    for _ in range(detector.history_length):
        detector.process(1000.0, np.full(NUM_BANDS, 1000.0))
        assert detector.get_global_onset() is False


def test_band_onsets_do_not_fire_during_warmup():
    detector = make_detector(window_duration=10 * BLOCK_DURATION)
    for _ in range(detector.history_length):
        detector.process(1000.0, np.full(NUM_BANDS, 1000.0))
        assert not detector.get_band_onsets().any()


def test_thresholds_stay_zero_during_warmup():
    detector = make_detector(window_duration=10 * BLOCK_DURATION)
    for _ in range(detector.history_length):
        detector.process(5.0, np.full(NUM_BANDS, 5.0))
        assert detector.get_global_threshold() == 0.0
        assert not detector.get_band_thresholds().any()


# ---- threshold math, independently recomputed ----

def test_global_threshold_matches_mean_plus_k_times_std_of_history():
    history_length = 8
    detector = make_detector(window_duration=history_length * BLOCK_DURATION, threshold_multiplier=2.0)
    values = np.arange(1, history_length + 1, dtype=float)
    warm_up(detector, values)

    detector.process(0.0, np.zeros(NUM_BANDS))

    expected = np.mean(values) + 2.0 * np.std(values)
    assert detector.get_global_threshold() == pytest.approx(expected)


def test_band_thresholds_match_mean_plus_k_times_std_per_band():
    history_length = 8
    detector = make_detector(window_duration=history_length * BLOCK_DURATION, threshold_multiplier=2.0)

    rng = np.random.default_rng(0)
    history = rng.uniform(0, 10, size=(history_length, NUM_BANDS))
    for row in history:
        detector.process(0.0, row)

    detector.process(0.0, np.zeros(NUM_BANDS))

    expected = np.mean(history, axis=0) + 2.0 * np.std(history, axis=0)
    assert detector.get_band_thresholds() == pytest.approx(expected)


# ---- onset triggering: rising edge + refractory ----

def test_global_onset_fires_once_on_spike_then_stays_silent_while_sustained():
    detector = make_detector(window_duration=20 * BLOCK_DURATION, threshold_multiplier=2.0, refractory_duration=2 * BLOCK_DURATION)
    warm_up(detector, [1.0] * detector.history_length)

    fired = []
    for _ in range(5):
        detector.process(1000.0, np.full(NUM_BANDS, 1000.0))
        fired.append(detector.get_global_onset())

    assert fired == [True, False, False, False, False]


def test_global_onset_refires_after_dropping_back_below_threshold():
    detector = make_detector(window_duration=20 * BLOCK_DURATION, threshold_multiplier=2.0, refractory_duration=1 * BLOCK_DURATION)
    warm_up(detector, [1.0] * detector.history_length)

    detector.process(1000.0, np.full(NUM_BANDS, 1000.0))
    assert detector.get_global_onset() is True

    for _ in range(detector.history_length):
        detector.process(1.0, np.full(NUM_BANDS, 1.0))

    detector.process(1000.0, np.full(NUM_BANDS, 1000.0))
    assert detector.get_global_onset() is True


def test_band_onset_fires_only_for_the_spiking_band():
    detector = make_detector(window_duration=20 * BLOCK_DURATION, threshold_multiplier=2.0)
    warm_up(detector, [1.0] * detector.history_length)

    spiked = np.full(NUM_BANDS, 1.0)
    spiked[3] = 1000.0
    detector.process(1.0, spiked)

    onsets = detector.get_band_onsets()
    assert onsets[3] is np.True_ or bool(onsets[3]) is True
    assert onsets.sum() == 1


def test_band_refractory_is_independent_per_band():
    detector = make_detector(window_duration=20 * BLOCK_DURATION, threshold_multiplier=2.0, refractory_duration=3 * BLOCK_DURATION)
    warm_up(detector, [1.0] * detector.history_length)

    spike_band_0 = np.full(NUM_BANDS, 1.0)
    spike_band_0[0] = 1000.0
    detector.process(1.0, spike_band_0)
    assert detector.get_band_onsets()[0]

    # band 0 is now in its refractory period, but band 1 spiking on the very
    # next block should be unaffected by it.
    spike_band_1 = np.full(NUM_BANDS, 1.0)
    spike_band_1[1] = 1000.0
    detector.process(1.0, spike_band_1)
    onsets = detector.get_band_onsets()
    assert onsets[1]
    assert not onsets[0]


def test_global_onset_refractory_blocks_immediate_retrigger():
    detector = make_detector(window_duration=20 * BLOCK_DURATION, threshold_multiplier=2.0, refractory_duration=5 * BLOCK_DURATION)
    warm_up(detector, [1.0] * detector.history_length)

    detector.process(1000.0, np.full(NUM_BANDS, 1000.0))
    assert detector.get_global_onset() is True

    detector.process(1.0, np.full(NUM_BANDS, 1.0))
    detector.process(1000.0, np.full(NUM_BANDS, 1000.0))
    assert detector.get_global_onset() is False


# ---- integration: real sine waves through FFTAnalyzer -> SpectralFlux -> OnsetDetection ----

@pytest.mark.parametrize("band_index", range(NUM_BANDS))
def test_band_onset_fires_for_a_tone_in_its_own_frequency_band(band_index):
    band = DEFAULT_FREQUENCY_BANDS.bands[band_index]
    frequency = bin_frequency_in_band(band)

    fft_analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    spectral_flux = SpectralFlux(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, hop_size=BLOCK_SIZE)
    onset_detection = OnsetDetection(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, window_duration=0.2)

    def feed(block):
        fft_analyzer.process(block)
        spectral_flux.process(fft_analyzer.get_raw_fft_data())
        onset_detection.process(spectral_flux.get_global_flux(), spectral_flux.get_band_fluxes())

    for _ in range(onset_detection.history_length + 2):
        feed(silence_block())

    feed(sine_block(frequency, amplitude=1.0))

    assert onset_detection.get_band_onsets()[band_index]
    assert onset_detection.get_global_onset()
