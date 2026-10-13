from pathlib import Path

from arlx.audio.fft_analyzer import DEFAULT_FREQUENCY_BANDS, FrequencyBands
import yaml

config_path = Path(__file__).resolve().parents[2] / "src" / "arlx" / "config" / "audio_config.yaml"

with open(config_path, 'r') as stream:
    dictionary = yaml.safe_load(stream)
print(dictionary)

loaded_bands = FrequencyBands.from_dict(dictionary["frequency_bands"])
print(loaded_bands)