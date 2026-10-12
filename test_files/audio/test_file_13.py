from pathlib import Path

import numpy as np

import dejavu.logic.decoder as decoder
from dejavu import Dejavu

from arlx.audio.song_detector import SongDetector

# Idealized test: no mic, no room, no air - loads a song file directly with
# dejavu's own decoder (the exact path fingerprint_directory uses) and feeds
# it through SongDetector in chunks, same as live capture would. Isolates
# "does the SongDetector code path work at all" from "is the mic signal good
# enough" (test_file_12's open question).
REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
RECORD_SECONDS = 5

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

if __name__ == "__main__":
    djv = Dejavu(config)
    song_detector = SongDetector(djv, record_seconds=RECORD_SECONDS, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)

    song_detector.fingerprint_directory(str(REPO_ROOT / "songs"), [".flac"])

    for song_path in sorted((REPO_ROOT / "songs").glob("*.flac")):
        channels, fs, _ = decoder.read(str(song_path))
        # back to float32 in [-1, 1] - the same format InputAudioCapture gives
        # SongDetector live, so this exercises the real add_samples() contract.
        samples_float32 = channels[0].astype(np.float32) / 32767.0

        print(f"\nFeeding {song_path.name} ({len(samples_float32)} samples)...")
        for start in range(0, len(samples_float32), BLOCK_SIZE):
            block = samples_float32[start:start + BLOCK_SIZE]
            if len(block) < BLOCK_SIZE:
                break
            song_detector.add_samples(block)

        results = song_detector.detect_song_from_blocks()
        print(f"Detected: {results}")
