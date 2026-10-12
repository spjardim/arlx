import time
from pathlib import Path

from dejavu import Dejavu

from arlx.audio.sources.input_capture import InputAudioCapture
from arlx.audio.song_detector import SongDetector

# Continuous version of test_file_11: instead of recording a fixed clip and
# recognizing it once, this keeps a rolling buffer (SongDetector) fed block by
# block and periodically asks it what it's hearing - same while-loop/process
# shape as test_file_6 through test_file_10, just with SongDetector standing
# in for the calculation class under test.
REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
RECORD_SECONDS = 5

# Full detection (spectrogram + db match) is expensive - only worth running a
# few times a second, not on every ~23ms block like the cheap calc classes.
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

    try:
        stream = InputAudioCapture(device_name="default", block_size=BLOCK_SIZE)
        stream.start()
        print("Press Ctrl+C to stop listening...")
        block_count = 0
        while True:
            returned, latest_sample, timestamp = stream.get_latest_sample_with_timestamp()
            if latest_sample is not None:
                song_detector.add_samples(latest_sample[:, 0].copy())
                block_count += 1

                if block_count % DETECT_EVERY_BLOCKS == 0:
                    results = song_detector.detect_song_from_blocks()
                    if results:
                        print(f"Heard: {results}")
                    else:
                        print("Still listening...")
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        stream.stop()
        print("Stopping the stream...")
