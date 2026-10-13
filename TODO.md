# TODO / Known Limitations

Tracking follow-up work surfaced while doing the getter/setter pass across
`AudioEngine` and the classes it constructs (`FFTAnalyzer`, `SpectralFlux`,
`OnsetDetection`, `TempoTracker`, `BeatPhaseTracker`, `DownbeatDetector`,
`SongDetector`, `SongLibrary`, `InputAudioCapture`). None of this is blocking
today's work - picking it off as it becomes a problem is fine.

## 1. Fingerprinting / precomputed feature data goes stale on parameter changes

- Changing `sample_rate` affects dejavu's fingerprint hashes - the live engine
  hashes incoming audio at the new rate, but previously-fingerprinted
  reference songs (and anything already matched) assumed the old rate.
- Separately, `SongLibrary._precompute_features` doesn't read the live
  `frequency_bands` at all - it hardcodes `DEFAULT_FREQUENCY_BANDS` and a
  literal `7` for `raw_band_energy`'s shape
  (`src/arlx/audio/song_library.py:80-81`). If `AudioEngine.set_frequency_bands`
  is ever called with different band boundaries/count, the live `FFTAnalyzer`
  and `SongLibrary`'s cached `song_features` table will silently diverge -
  same band names, different frequency boundaries.
- Need to decide: recompute/invalidate stored features on certain parameter
  changes, or keep `song_features` as a fixed schema independent of live
  config (and document that explicitly).

## 2. Config write-back / persistence lifecycle

- Every setter so far only mutates in-memory instance state. Nothing writes
  the change back into `self.dictionary`, and nothing persists
  `self.dictionary` back to `audio_config.yaml`.
- Need a plan for: which setters should update the dictionary, when/how the
  dictionary gets saved (on every set? an explicit `save_config()`? on exit?),
  and whether partial/invalid state can get saved.

## 3. Test suite needs a full pass

- 42 currently-failing tests, all traceable to two known causes:
  - Stale kwargs in test fixtures: `hop_size` (removed from `SpectralFlux`)
    and `frame_size` (renamed to `block_size`).
  - `tests/audio/test_fft_analyzer.py::test_more_than_seven_bands_raises_index_error`
    pins the old hardcoded-7-bins bug, which has since been deliberately
    fixed - needs rewriting to assert the new correct behavior instead.
- Plus: none of the new getters/setters across any class have test coverage
  yet.

## 4. Live audio-stream lifecycle gap

- `InputAudioCapture.set_block_size` / `set_device_name` only update
  attributes. The actual `sd.InputStream` was constructed with a fixed
  `blocksize`/device at `start()` time, and sounddevice has no
  live-reconfigure API.
- Calling `AudioEngine.set_block_size(...)` (or changing device) while the
  stream is actively running updates every processing class's internal state
  but does **not** change what's actually coming off the hardware until
  something calls `stop()` + `start()` again. Nothing does that
  automatically today.

## 5. No input validation on setters

- Nothing stops e.g. `set_min_bpm(200)` with `max_bpm=70` (inverted range),
  `set_block_size(0)`, `set_beats_per_bar(0)`, negative sample rates, etc.
  Decide how defensive these need to be.

## 6. No thread-safety story for the new setters

- `InputAudioCapture` protects `latest_sample`/`latest_timestamp` with a lock
  because the sounddevice callback runs on its own thread. None of the new
  setters (on `FFTAnalyzer`, `SpectralFlux`, `OnsetDetection`, `TempoTracker`,
  `BeatPhaseTracker`, `DownbeatDetector`, `SongDetector`) have any such
  protection - calling one while `process()` is mid-flight on another thread
  is a race (e.g. resizing an array mid-read).

## 7. Loose thread: `get_audio_input_sample_rate`

- `src/arlx/audio/audio_engine.py` still has a `# NOTE: Do we need this?`
  above `get_audio_input_sample_rate` - unresolved, predates the items above,
  easy to lose track of.
