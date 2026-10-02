import numpy as np
import pytest

from arlx.audio.raw_audio_feature_extractor import RawAudioFeatureExtractor

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024


def sine_block(frequency: float, amplitude: float = 1.0, n=BLOCK_SIZE, sample_rate=SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def silence_block(n=BLOCK_SIZE) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def test_rms_matches_sine_amplitude():
    extractor = RawAudioFeatureExtractor(SAMPLE_RATE, BLOCK_SIZE)
    features = extractor.extract(sine_block(1000, amplitude=1.0))
    assert features.rms == pytest.approx(1 / np.sqrt(2), abs=0.01)


def test_band_energy_peaks_in_expected_band():
    extractor = RawAudioFeatureExtractor(SAMPLE_RATE, BLOCK_SIZE)
    features = extractor.extract(sine_block(1000))
    loudest_band = max(features.band_energy, key=features.band_energy.get)
    assert loudest_band == "mid"


def test_spectral_centroid_tracks_tone_frequency():
    extractor = RawAudioFeatureExtractor(SAMPLE_RATE, BLOCK_SIZE)
    features = extractor.extract(sine_block(1000))
    assert features.spectral_centroid == pytest.approx(1000, abs=100)


def test_spectral_rolloff_sits_near_pure_tone_frequency():
    extractor = RawAudioFeatureExtractor(SAMPLE_RATE, BLOCK_SIZE)
    features = extractor.extract(sine_block(1000))
    # almost all the energy is in the 1000Hz bin, so the 85% rolloff point
    # should land right around it rather than trailing off far above.
    assert features.spectral_rolloff == pytest.approx(1000, abs=100)


def test_spectral_flatness_is_low_for_pure_tone_and_high_for_noise():
    extractor = RawAudioFeatureExtractor(SAMPLE_RATE, BLOCK_SIZE)
    tone_features = extractor.extract(sine_block(1000))

    rng = np.random.default_rng(0)
    noise_block = rng.uniform(-1.0, 1.0, BLOCK_SIZE).astype(np.float32)
    noise_features = extractor.extract(noise_block)

    assert tone_features.spectral_flatness < 0.1
    assert noise_features.spectral_flatness > 0.5


def test_onset_fires_when_sound_follows_silence():
    extractor = RawAudioFeatureExtractor(SAMPLE_RATE, BLOCK_SIZE)
    for _ in range(5):
        extractor.extract(silence_block())

    features = extractor.extract(sine_block(1000))
    assert features.onset is True


def test_onset_does_not_fire_on_sustained_steady_tone():
    extractor = RawAudioFeatureExtractor(SAMPLE_RATE, BLOCK_SIZE)
    extractor.extract(sine_block(1000))

    onsets = [extractor.extract(sine_block(1000)).onset for _ in range(10)]
    assert not any(onsets)
