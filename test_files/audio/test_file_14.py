from pathlib import Path

import numpy as np
import sounddevice as sd

import dejavu.logic.decoder as decoder
from dejavu import Dejavu

from arlx.audio.song_detector import SongDetector

# Plays a song out loud through the speakers while feeding the exact same
# unaltered blocks into SongDetector, in real time (OutputStream.write()
# blocks until playback has room, which paces the loop naturally). No mic, no
# air, no room - isolates "does streaming detection work over real time" from
# both the code-bug question (test_file_13 already answered that) and the
# live-mic signal-quality question (test_file_12's open question).
REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
RECORD_SECONDS = 5

DETECT_EVERY_SECONDS = 1.0
DETECT_EVERY_BLOCKS = int(DETECT_EVERY_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

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
        samples_float32 = channels[0].astype(np.float32) / 32767.0

        print(f"\nPlaying {song_path.name} ({len(samples_float32) / fs:.1f}s)...")
        stream = sd.OutputStream(samplerate=fs, channels=1, dtype="float32", blocksize=BLOCK_SIZE)
        stream.start()
        try:
            block_count = 0
            for start in range(0, len(samples_float32), BLOCK_SIZE):
                block = samples_float32[start:start + BLOCK_SIZE]
                if len(block) < BLOCK_SIZE:
                    break

                stream.write(block)  # blocks until there's room - paces the loop to real time
                song_detector.add_samples(block)
                block_count += 1

                if block_count % DETECT_EVERY_BLOCKS == 0:
                    results = song_detector.detect_song_from_blocks()
                    if results:
                        print(f"Heard: {results}")
                        break  # stop playback early once confirmed, don't sit through the whole track
        finally:
            stream.stop()
            stream.close()
