from pathlib import Path

from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer

# First smoke test of the dejavu submodule (third_party/dejavu). Fingerprints
# the sample mp3s dejavu ships with itself and tries to recognize one of the
# short wav clips also bundled there - validates the whole install (ffmpeg,
# Postgres, the forked/patched package) before pointing it at a real library.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEJAVU_ROOT = REPO_ROOT / "third_party" / "dejavu"

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

    djv.fingerprint_directory(str(DEJAVU_ROOT / "mp3"), [".mp3"])

    test_clip = DEJAVU_ROOT / "test" / "woodward_43s.wav"
    results = djv.recognize(FileRecognizer, str(test_clip))
    print(f"Recognized: {results}")
