from pathlib import Path

from arlx.audio.sources.input_capture import InputAudioCapture
from arlx.audio.fft_analyzer import FFTAnalyzer, FrequencyBands
from arlx.audio.spectral_flux import SpectralFlux
from arlx.audio.onest_detection import OnsetDetection
from arlx.audio.tempo_tracker import TempoTracker
from arlx.audio.beat_phase_tracker import BeatPhaseTracker
from arlx.audio.downbeat_detector import DownbeatDetector
from arlx.audio.song_detector import SongDetector
from arlx.audio.song_library import SongLibrary
from dejavu import Dejavu
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

# NOTE: updated some parameters requires recomputing saved song data
# NOTE: Need to add in config adjustments for setting
# NOTE: Need to full update test suites
class AudioEngine():

    def __init__(self):

        config_path = Path(__file__).resolve().parent.parent / "config" / "audio_config.yaml"
        with open(config_path, 'r') as stream:
            self.dictionary = yaml.safe_load(stream)
        self.block_size = self.dictionary["audio_stream"]["block_size"]

        self.audio_input = InputAudioCapture(
            device_name="default", # Just for the sake of initializtion pick a name we know is there
            block_size=self.block_size
        )

        self.sample_rate = self.audio_input.get_input_stream_sample_rate()

        self.frequency_bands = FrequencyBands.from_dict(self.dictionary["frequency_bands"])

        self.fft_analyzer = FFTAnalyzer(
            self.sample_rate,
            self.block_size,
            self.frequency_bands,
            **self.dictionary["fft_analyzer"]
        )

        self.spectral_flux = SpectralFlux(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            frequency_bands=self.frequency_bands,
            **self.dictionary["spectral_flux"]

        )

        self.onset_detection = OnsetDetection(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            frequency_bands=self.frequency_bands,
            **self.dictionary["onset_detection"]
        )

        self.tempo_tracker = TempoTracker(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            **self.dictionary["tempo_tracker"]
        )

        self.beat_phase_tracker = BeatPhaseTracker(
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            **self.dictionary["beat_phase_tracker"]
        )

        self.downbeat_detector = DownbeatDetector(
            **self.dictionary["downbeat_detector"]
        )

        self.dejavu = Dejavu(self.dictionary["dejavu"])

        self.song_detector = SongDetector(
            self.dejavu,
            sample_rate=self.sample_rate,
            block_size=self.block_size,
            **self.dictionary["song_detector"]
        )
        self.song_detector.fingerprint_directory(
            str(REPO_ROOT / self.dictionary["songs"]["directory"]),
            self.dictionary["songs"]["extensions"]
        )

        self.song_library = SongLibrary(self.dejavu, self.dictionary["dejavu"]["database"])

    ### Engine Getters/Setters ###
    def get_audio_block_size(self):
        return self.block_size

    def set_block_size(self, block_size: int):
        self.block_size = block_size
        self.audio_input.set_block_size(block_size)
        self.fft_analyzer.set_block_size(block_size)
        self.spectral_flux.set_block_size(block_size)
        self.onset_detection.set_block_size(block_size)
        self.tempo_tracker.set_block_size(block_size)
        self.beat_phase_tracker.set_block_size(block_size)
        self.song_detector.set_block_size(block_size)

    def get_audio_sample_rate(self):
        return self.sample_rate

    def _set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.fft_analyzer.set_sample_rate(sample_rate)
        self.spectral_flux.set_sample_rate(sample_rate)
        self.onset_detection.set_sample_rate(sample_rate)
        self.tempo_tracker.set_sample_rate(sample_rate)
        self.beat_phase_tracker.set_sample_rate(sample_rate)
        self.song_detector.set_sample_rate(sample_rate)
        
    def get_frequency_bands(self):
        return self.frequency_bands

    def set_frequency_bands(self, frequency_bands: FrequencyBands):
        self.frequency_bands = frequency_bands
        self.fft_analyzer.set_frequency_bands(frequency_bands)
        self.spectral_flux.set_frequency_bands(frequency_bands)
        self.onset_detection.set_frequency_bands(frequency_bands)

    ### Audio Input Getters/Setters ###
    def get_audio_devices(self):
        return self.audio_input.get_potential_input_devices()

    def get_audio_input_block_size(self):
        return self.audio_input.get_block_size()

    def get_audio_device_name(self):
        return self.audio_input.get_device_name()

    def set_audio_device_name(self, device_name: str):
        self.audio_input.set_device_name(device_name)

    def is_audio_device_connected(self):
        return self.audio_input.is_stream_connected()

    # NOTE: Do we need this?
    def get_audio_input_sample_rate(self):
        return self.audio_input.get_input_stream_sample_rate()

    ### FFTAnalyzer Getters/Setters ###
    def get_fft_block_size(self):
        return self.fft_analyzer.get_block_size()

    def get_fft_sample_rate(self):
        return self.fft_analyzer.get_sample_rate()

    def get_fft_frequency_bands(self):
        return self.fft_analyzer.get_frequency_bands()

    def get_fft_smooth_attack(self):
        return self.fft_analyzer.get_smooth_attack()

    def set_fft_smooth_attack(self, smooth_attack: float):
        self.fft_analyzer.set_smooth_attack(smooth_attack)

    def get_fft_smooth_decay(self):
        return self.fft_analyzer.get_smooth_decay()

    def set_fft_smooth_decay(self, smooth_decay: float):
        self.fft_analyzer.set_smooth_decay(smooth_decay)

    def get_fft_attack_db_per_block(self):
        return self.fft_analyzer.get_attack_db_per_block()

    def set_fft_attack_db_per_block(self, attack_db_per_block: float):
        self.fft_analyzer.set_attack_db_per_block(attack_db_per_block)

    def get_fft_decay_db_per_block(self):
        return self.fft_analyzer.get_decay_db_per_block()

    def set_fft_decay_db_per_block(self, decay_db_per_block: float):
        self.fft_analyzer.set_decay_db_per_block(decay_db_per_block)

    def get_fft_floor_db(self):
        return self.fft_analyzer.get_floor_db()

    def set_fft_floor_db(self, floor_db: float):
        self.fft_analyzer.set_floor_db(floor_db)

    def get_fft_use_window(self):
        return self.fft_analyzer.get_use_window()

    def set_fft_use_window(self, use_window: bool):
        self.fft_analyzer.set_use_window(use_window)

    ### SpectralFlux Getters/Setters ###
    def get_spectral_flux_block_size(self):
        return self.spectral_flux.get_block_size()

    def get_spectral_flux_sample_rate(self):
        return self.spectral_flux.get_sample_rate()

    def get_spectral_flux_frequency_bands(self):
        return self.spectral_flux.get_frequency_bands()

    def get_spectral_flux_half_wave_rectify(self):
        return self.spectral_flux.get_half_wave_rectify()

    def set_spectral_flux_half_wave_rectify(self, half_wave_rectify: bool):
        self.spectral_flux.set_half_wave_rectify(half_wave_rectify)

    def get_spectral_flux_use_log_magnitude(self):
        return self.spectral_flux.get_use_log_magnitude()

    def set_spectral_flux_use_log_magnitude(self, use_log_magnitude: bool):
        self.spectral_flux.set_use_log_magnitude(use_log_magnitude)

    def get_spectral_flux_normalize_by_bin_count(self):
        return self.spectral_flux.get_normalize_by_bin_count()

    def set_spectral_flux_normalize_by_bin_count(self, normalize_by_bin_count: bool):
        self.spectral_flux.set_normalize_by_bin_count(normalize_by_bin_count)

    def get_spectral_flux_log_epsilon(self):
        return self.spectral_flux.get_log_epsilon()

    def set_spectral_flux_log_epsilon(self, log_epsilon: float):
        self.spectral_flux.set_log_epsilon(log_epsilon)

    ### OnsetDetection Getters/Setters ###
    def get_onset_detection_block_size(self):
        return self.onset_detection.get_block_size()

    def get_onset_detection_sample_rate(self):
        return self.onset_detection.get_sample_rate()

    def get_onset_detection_frequency_bands(self):
        return self.onset_detection.get_frequency_bands()

    def get_onset_detection_window_duration(self):
        return self.onset_detection.get_window_duration()

    def set_onset_detection_window_duration(self, window_duration: float):
        self.onset_detection.set_window_duration(window_duration)

    def get_onset_detection_threshold_multiplier(self):
        return self.onset_detection.get_threshold_multiplier()

    def set_onset_detection_threshold_multiplier(self, threshold_multiplier: float):
        self.onset_detection.set_threshold_multiplier(threshold_multiplier)

    def get_onset_detection_refractory_duration(self):
        return self.onset_detection.get_refractory_duration()

    def set_onset_detection_refractory_duration(self, refractory_duration: float):
        self.onset_detection.set_refractory_duration(refractory_duration)

    ### TempoTracker Getters/Setters ###
    def get_tempo_tracker_block_size(self):
        return self.tempo_tracker.get_block_size()

    def get_tempo_tracker_sample_rate(self):
        return self.tempo_tracker.get_sample_rate()

    def get_tempo_tracker_min_bpm(self):
        return self.tempo_tracker.get_min_bpm()

    def set_tempo_tracker_min_bpm(self, min_bpm: float):
        self.tempo_tracker.set_min_bpm(min_bpm)

    def get_tempo_tracker_max_bpm(self):
        return self.tempo_tracker.get_max_bpm()

    def set_tempo_tracker_max_bpm(self, max_bpm: float):
        self.tempo_tracker.set_max_bpm(max_bpm)

    def get_tempo_tracker_history_duration(self):
        return self.tempo_tracker.get_history_duration()

    def set_tempo_tracker_history_duration(self, history_duration: float):
        self.tempo_tracker.set_history_duration(history_duration)

    def get_tempo_tracker_smoothing(self):
        return self.tempo_tracker.get_smoothing()

    def set_tempo_tracker_smoothing(self, smoothing: float):
        self.tempo_tracker.set_smoothing(smoothing)

    ### BeatPhaseTracker Getters/Setters ###
    def get_beat_phase_tracker_block_size(self):
        return self.beat_phase_tracker.get_block_size()

    def get_beat_phase_tracker_sample_rate(self):
        return self.beat_phase_tracker.get_sample_rate()

    def get_beat_phase_tracker_correction_strength(self):
        return self.beat_phase_tracker.get_correction_strength()

    def set_beat_phase_tracker_correction_strength(self, correction_strength: float):
        self.beat_phase_tracker.set_correction_strength(correction_strength)

    def get_beat_phase_tracker_beat_tolerance(self):
        return self.beat_phase_tracker.get_beat_tolerance()

    def set_beat_phase_tracker_beat_tolerance(self, beat_tolerance: float):
        self.beat_phase_tracker.set_beat_tolerance(beat_tolerance)

    ### DownbeatDetector Getters/Setters ###
    def get_downbeat_detector_beats_per_bar(self):
        return self.downbeat_detector.get_beats_per_bar()

    def set_downbeat_detector_beats_per_bar(self, beats_per_bar: int):
        self.downbeat_detector.set_beats_per_bar(beats_per_bar)

    def get_downbeat_detector_history_bars(self):
        return self.downbeat_detector.get_history_bars()

    def set_downbeat_detector_history_bars(self, history_bars: int):
        self.downbeat_detector.set_history_bars(history_bars)

    ### SongDetector Getters/Setters ###
    def get_song_detector_block_size(self):
        return self.song_detector.get_block_size()

    def get_song_detector_sample_rate(self):
        return self.song_detector.get_sample_rate()

    def get_song_detector_record_seconds(self):
        return self.song_detector.get_record_seconds()

    def set_song_detector_record_seconds(self, record_seconds: float):
        self.song_detector.set_record_seconds(record_seconds)

    def get_song_detector_detect_every_seconds(self):
        return self.song_detector.get_detect_every_seconds()

    def set_song_detector_detect_every_seconds(self, detect_every_seconds: float):
        self.song_detector.set_detect_every_seconds(detect_every_seconds)

    def get_song_detector_sync_check_seconds(self):
        return self.song_detector.get_sync_check_seconds()

    def set_song_detector_sync_check_seconds(self, sync_check_seconds: float):
        self.song_detector.set_sync_check_seconds(sync_check_seconds)

    def get_song_detector_sync_check_every_seconds(self):
        return self.song_detector.get_sync_check_every_seconds()

    def set_song_detector_sync_check_every_seconds(self, sync_check_every_seconds: float):
        self.song_detector.set_sync_check_every_seconds(sync_check_every_seconds)

    def get_song_detector_sync_confidence_threshold(self):
        return self.song_detector.get_sync_confidence_threshold()

    def set_song_detector_sync_confidence_threshold(self, sync_confidence_threshold: float):
        self.song_detector.set_sync_confidence_threshold(sync_confidence_threshold)

    ### SongLibrary Getters/Setters ###
    def get_song_library_block_size(self):
        return self.song_library.get_block_size()

    def set_song_library_block_size(self, block_size: int):
        self.song_library.set_block_size(block_size)

if __name__ == "__main__":
    engine = AudioEngine()