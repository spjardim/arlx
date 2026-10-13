import numpy as np
import pytest

from arlx.audio.fft_analyzer import DEFAULT_FREQUENCY_BANDS, FFTAnalyzer, FrequencyBand, FrequencyBands
from arlx.audio.spectral_flux import SpectralFlux

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

# four single-Hz-wide bands over a tiny 4-bin spectrum (sample_rate chosen so
# fftfreq lands on clean integers 0,1,2,3), used for hand-checkable band math.
TINY_SAMPLE_RATE = 8
TINY_FRAME_SIZE = 8
SINGLE_BIN_BANDS = FrequencyBands([
    FrequencyBand("b0", 0, 1),
    FrequencyBand("b1", 1, 2),
    FrequencyBand("b2", 2, 3),
    FrequencyBand("b3", 3, 4),
])


def sine_block(frequency: float, amplitude: float = 1.0, n=BLOCK_SIZE, sample_rate=SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def bin_frequency_in_band(band, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE) -> float:
    spec_x = np.fft.fftfreq(block_size, d=1.0 / sample_rate)[:block_size // 2]
    in_band = spec_x[(spec_x >= band.low_freq) & (spec_x < band.high_freq)]
    assert len(in_band) > 0, f"no FFT bin falls inside band '{band.name}'"
    return float(in_band[len(in_band) // 2])


def make_tiny_flux(**overrides):
    params = dict(
        sample_rate=TINY_SAMPLE_RATE,
        frame_size=TINY_FRAME_SIZE,
        hop_size=TINY_FRAME_SIZE,
        frequency_bands=SINGLE_BIN_BANDS,
    )
    params.update(overrides)
    return SpectralFlux(**params)


# ---- first call / baseline ----

def test_first_call_produces_zero_global_and_band_flux():
    flux = make_tiny_flux()
    flux.process(np.array([1.0, 2.0, 3.0, 4.0]))
    assert flux.get_global_flux() == 0.0
    assert not flux.get_band_fluxes().any()


# ---- log-domain diff matches manual calculation ----

def test_global_flux_matches_manual_log_domain_calculation():
    flux = make_tiny_flux()
    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    flux.process(np.array([1.0, 2.0, 1.0, 1.0]))

    expected_diff = 20 * np.log10([1.0, 2.0, 1.0, 1.0]) - 20 * np.log10([1.0, 1.0, 1.0, 1.0])
    expected_diff = np.maximum(expected_diff, 0)
    expected_global = np.mean(expected_diff ** 2)

    assert flux.get_global_flux() == pytest.approx(expected_global)


def test_band_fluxes_isolate_the_change_to_the_correct_bin():
    flux = make_tiny_flux()
    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    flux.process(np.array([1.0, 2.0, 1.0, 1.0]))

    band_fluxes = flux.get_band_fluxes()
    assert band_fluxes[1] > 0
    assert band_fluxes[0] == 0
    assert band_fluxes[2] == 0
    assert band_fluxes[3] == 0


# ---- half-wave rectification ----

def test_half_wave_rectification_zeroes_out_a_decrease():
    flux = make_tiny_flux(half_wave_rectify=True)
    flux.process(np.array([2.0, 2.0, 2.0, 2.0]))
    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    assert flux.get_global_flux() == 0.0


def test_disabling_half_wave_rectification_lets_a_decrease_contribute():
    flux = make_tiny_flux(half_wave_rectify=False)
    flux.process(np.array([2.0, 2.0, 2.0, 2.0]))
    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    assert flux.get_global_flux() > 0.0


# ---- linear vs log magnitude ----

def test_linear_magnitude_uses_raw_difference_not_log():
    flux = make_tiny_flux(use_log_magnitude=False)
    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    flux.process(np.array([1.0, 3.0, 1.0, 1.0]))

    expected = np.mean(np.maximum(np.array([1.0, 3.0, 1.0, 1.0]) - np.array([1.0, 1.0, 1.0, 1.0]), 0) ** 2)
    assert flux.get_global_flux() == pytest.approx(expected)


# ---- sum vs mean reduction ----

def test_normalize_by_bin_count_false_sums_instead_of_averaging():
    summed = make_tiny_flux(use_log_magnitude=False, normalize_by_bin_count=False)
    averaged = make_tiny_flux(use_log_magnitude=False, normalize_by_bin_count=True)

    for flux in (summed, averaged):
        flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
        flux.process(np.array([1.0, 3.0, 2.0, 1.0]))

    assert summed.get_global_flux() == pytest.approx(averaged.get_global_flux() * 4)


# ---- bands with no bins in range ----

def test_band_with_no_bins_in_range_reports_zero_flux():
    empty_band = FrequencyBands([FrequencyBand("none", 10, 20)])
    flux = SpectralFlux(sample_rate=TINY_SAMPLE_RATE, block_size=TINY_FRAME_SIZE, hop_size=TINY_FRAME_SIZE, frequency_bands=empty_band)

    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    flux.process(np.array([1.0, 5.0, 1.0, 1.0]))

    assert flux.get_band_fluxes()[0] == 0.0


# ---- accessors ----

def test_get_band_fluxes_returns_a_copy():
    flux = make_tiny_flux()
    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    flux.process(np.array([1.0, 2.0, 1.0, 1.0]))

    band_fluxes = flux.get_band_fluxes()
    band_fluxes[:] = -999
    assert not np.any(flux.get_band_fluxes() == -999)


def test_update_frequency_bands_changes_band_count_on_next_process():
    flux = make_tiny_flux()
    flux.process(np.array([1.0, 1.0, 1.0, 1.0]))
    assert len(flux.get_band_fluxes()) == 4

    flux.update_frequency_bands(FrequencyBands([FrequencyBand("only", 0, 4)]))
    flux.process(np.array([1.0, 2.0, 1.0, 1.0]))
    assert len(flux.get_band_fluxes()) == 1


# ---- integration: real sine waves through FFTAnalyzer ----

@pytest.mark.parametrize("band_index", range(7))
def test_band_flux_spikes_in_the_band_matching_a_new_tone(band_index):
    band = DEFAULT_FREQUENCY_BANDS.bands[band_index]
    frequency = bin_frequency_in_band(band)

    analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    flux = SpectralFlux(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, hop_size=BLOCK_SIZE)

    analyzer.process(np.zeros(BLOCK_SIZE, dtype=np.float32))
    flux.process(analyzer.get_raw_fft_data())

    analyzer.process(sine_block(frequency, amplitude=1.0))
    flux.process(analyzer.get_raw_fft_data())

    assert np.argmax(flux.get_band_fluxes()) == band_index
