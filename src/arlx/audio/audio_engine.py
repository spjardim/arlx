
from typing import Optional

from arlx.audio.audio_feature_tiers import AudioFeatures
from arlx.audio.raw_audio_feature_extractor import RawAudioFeatureExtractor
from arlx.audio.sources.base import AudioSource


class AudioEngine():

    def __init__(self, audio_source: Optional[AudioSource] = None):
        self.audio_source: AudioSource = audio_source
        self.audio_feature_engine = RawAudioFeatureExtractor(sample_rate=44100, block_size=1024)

    def update_audio_source(self, audio_source: AudioSource):
        self.audio_source = audio_source

    def get_audio_features(self):
        if self.audio_source is None:
            return None
        audio = self.audio_source.get_latest_sample()
        if audio is None:
            return None
        features = self.audio_feature_engine.extract(audio)
        return AudioFeatures(tier0=features)