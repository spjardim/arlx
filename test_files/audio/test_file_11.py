import time
from pathlib import Path

import numpy as np

from dejavu import Dejavu

from arlx.audio.sources.input_capture import InputAudioCapture

REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
RECORD_SECONDS = 10

config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "password",
        "database": "dejavu",
    },
    "database_type": "postgres",
}


def record_seconds(seconds: float) -> np.ndarray:
    stream = InputAudioCapture(device_name="default", block_size=BLOCK_SIZE)
    stream.start()
    blocks = []
    blocks_needed = int(seconds * SAMPLE_RATE / BLOCK_SIZE)
    print(f"Listening for {seconds} seconds...")
    try:
        while len(blocks) < blocks_needed:
            returned, latest_sample, timestamp = stream.get_latest_sample_with_timestamp()
            if latest_sample is not None:
                blocks.append(latest_sample[:, 0].copy())
            else:
                time.sleep(0.001)
    finally:
        stream.stop()
    return np.concatenate(blocks)


if __name__ == "__main__":
    djv = Dejavu(config)

    djv.fingerprint1_directory(str(REPO_ROOT / "songs"), [".flac"])

    samples = record_seconds(RECORD_SECONDS)
    # dejavu expects int16-scale PCM, same as what it reads from files -
    # sounddevice gives float32 in [-1, 1], so rescale before fingerprinting.
    samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)

    hashes, fingerprint_time = djv.generate_fingerprints(samples_int16, Fs=SAMPLE_RATE)
    matches, dedup_hashes, query_time = djv.find_matches(hashes)
    results = djv.align_matches(matches, dedup_hashes, len(hashes))

    print(f"Heard: {results}")
