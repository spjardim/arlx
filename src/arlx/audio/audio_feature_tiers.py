

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AudioTiers(Enum):
    RAW = 1
    SONG_AWARE = 2
    DJ_LINK = 3

@dataclass
class RawAudioBands:
    bass: float
    low_mid: float
    mid: float
    high_mid: float
    treble: float

    def __str__(self) -> str:
        return (
            f"RawAudioBands(bass={self.bass:.3f}, low_mid={self.low_mid:.3f}, "
            f"mid={self.mid:.3f}, high_mid={self.high_mid:.3f}, treble={self.treble:.3f})"
        )
    
@dataclass
class RawAudioFeatures:
    rms: float
    peak: float
    band_energy: RawAudioBands
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flatness: float
    spectral_flux: float
    onset: bool

    def __str__(self) -> str:
        return (
            f"RawAudioFeatures(rms={self.rms:.3f}, peak={self.peak:.3f}, "
            f"band_energy={self.band_energy}, spectral_centroid={self.spectral_centroid:.1f}, "
            f"spectral_rolloff={self.spectral_rolloff:.1f}, spectral_flatness={self.spectral_flatness:.3f}, "
            f"spectral_flux={self.spectral_flux:.3f}, onset={self.onset})"
        )

@dataclass
class SongAwareFeatures:
    """
    Song identity, populated on a fingerprint cache hit.

    sync_offset_sec is "how far into the matched track we currently are";
    combined with event_timeline, effects can compute time-until-next-event
    for anticipatory behavior (e.g. pre-staging a build-up before a drop).
    """
    track_id: Optional[str] = None
    confidence: float = 0.0
    sync_offset_sec: float = 0.0
    # Ordered list of (offset_sec, event_label) tuples, e.g. (63.5, "drop").
    # Populated from the song cache; empty if the track has never been
    # background-analyzed even though it was identified.
    event_timeline: list = field(default_factory=list)

@dataclass
class DJLinkFeatures:
    """
    Features derived from DJ Link metadata, e.g. BPM, key, etc.
    """
    bpm: Optional[float] = None
    key: Optional[str] = None


@dataclass
class AudioFeatures:
    """
    One frame's full audio snapshot, merged from every active AudioSource.

    tier0 is never None. tier1 / tier2 are None when no source populated
    them this frame (no fingerprint match yet, no DJ hardware connected).
    """
    tier0: RawAudioFeatures
    tier1: Optional[SongAwareFeatures] = None
    tier2: Optional[DJLinkFeatures] = None

    @property
    def active_tiers(self) -> set:
        tiers = {AudioTiers.RAW}
        if self.tier1 is not None:
            tiers.add(AudioTiers.SONG_AWARE)
        if self.tier2 is not None:
            tiers.add(AudioTiers.DJ_LINK)
        return tiers