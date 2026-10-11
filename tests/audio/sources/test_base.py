import numpy as np
import pytest

from arlx.audio.sources.base import AudioSource


def test_audio_source_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AudioSource()


def test_subclass_missing_an_abstract_method_cannot_be_instantiated():
    class IncompleteSource(AudioSource):
        def start(self) -> bool:
            return True

        def stop(self) -> bool:
            return True

        def get_latest_sample(self) -> np.ndarray:
            return np.zeros(1)
        # get_latest_sample_with_timestamp intentionally left unimplemented

    with pytest.raises(TypeError):
        IncompleteSource()


def test_subclass_implementing_all_abstract_methods_can_be_instantiated():
    class CompleteSource(AudioSource):
        def start(self) -> bool:
            return True

        def stop(self) -> bool:
            return True

        def get_latest_sample(self) -> np.ndarray:
            return np.zeros(1)

        def get_latest_sample_with_timestamp(self):
            return True, np.zeros(1), 0.0

    source = CompleteSource()
    assert source.start() is True
    assert source.stop() is True
