# Where each frequency band sits on the hue wheel (0-1). Kept inside
# [0, 0.85] rather than spanning the full 0-1 circle so a treble-dominant
# mix doesn't wrap back around to red and look indistinguishable from a
# bass-dominant one.
BAND_HUES = {
    "bass": 0.0,
    "low_mid": 0.2,
    "mid": 0.45,
    "high_mid": 0.65,
    "treble": 0.85,
}
