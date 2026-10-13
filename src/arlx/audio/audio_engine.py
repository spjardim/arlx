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

    def list_potential_input_devices(self) -> list[str]:
        return self.audio_input.list_potential_input_devices()

    def set_input_device(self, device_name: str = "default"):
        self.audio_input = InputAudioCapture(device_name=device_name, block_size=1024)
        self.sample_rate = self.audio_input.get_input_stream_sample_rate()
        self.block_size = self.audio_input.block_size