"""Global sample/slot/half-hour indices, the block grids they're summed onto,
and a run's `Cadence` (slot length and tau ladder), from which every slot,
window and block size derives.
"""
from dataclasses import dataclass
from math import gcd

import numpy as np
import pandas as pd

from ttu_tower.constants import ROWS_PER_FILE, SAMPLE_HZ

EPOCH = pd.Timestamp("2000-01-01 00:00", tz="Etc/GMT+6")

_SAMPLE_NS = 1_000_000_000 // SAMPLE_HZ  # 20 ms
SAMPLES_PER_SLOT = 10 * 60 * SAMPLE_HZ  # 30_000 at the default cadence; pipeline code uses Cadence.samples_per_slot
SAMPLES_PER_HALF_HOUR = ROWS_PER_FILE  # 90_000 == 3 * SAMPLES_PER_SLOT at the default cadence


def _require_tz_aware(t) -> None:
    tz = t.tzinfo if isinstance(t, pd.Timestamp) else t.tz
    if tz is None:
        raise ValueError(f"{t} is not tz-aware")


def time_to_sample(t: pd.Timestamp | pd.DatetimeIndex):
    """Timestamp(s) -> sample index g (int64). t must fall exactly on a 20 ms tick."""
    _require_tz_aware(t)
    if isinstance(t, pd.Timestamp):
        delta_ns = t.value - EPOCH.value
        g, rem = divmod(delta_ns, _SAMPLE_NS)
        if rem != 0:
            raise ValueError(f"{t} is not aligned to a sample (20 ms) tick")
        return int(g)
    idx = pd.DatetimeIndex(t)
    delta_ns = idx.asi8 - EPOCH.value
    g, rem = np.divmod(delta_ns, _SAMPLE_NS)
    if np.any(rem != 0):
        raise ValueError("some timestamps are not aligned to a sample (20 ms) tick")
    return g.astype(np.int64)


def sample_to_time(g) -> pd.Timestamp | pd.DatetimeIndex:
    """Sample index g -> Timestamp(s), tz-aware in EPOCH's zone (Etc/GMT+6)."""
    scalar = np.isscalar(g)
    g_arr = np.atleast_1d(np.asarray(g, dtype=np.int64))
    ns = EPOCH.value + g_arr * _SAMPLE_NS
    idx = pd.DatetimeIndex(ns, tz="UTC").tz_convert(EPOCH.tz)
    return idx[0] if scalar else idx


def _index_to_time(index, block_samples: int):
    return sample_to_time(np.asarray(index) * block_samples if not np.isscalar(index) else index * block_samples)


def _time_to_index(t, block_samples: int, name: str):
    g = time_to_sample(t)
    if np.isscalar(g):
        if g % block_samples != 0:
            raise ValueError(f"{t} is not aligned to a {name} boundary")
        return g // block_samples
    if np.any(g % block_samples != 0):
        raise ValueError(f"some timestamps are not aligned to a {name} boundary")
    return g // block_samples


def slot_to_time(k) -> pd.Timestamp | pd.DatetimeIndex:
    """Slot index k -> its start time."""
    return _index_to_time(k, SAMPLES_PER_SLOT)


def time_to_slot(t):
    """Time -> slot index k. t must fall exactly on a slot boundary."""
    return _time_to_index(t, SAMPLES_PER_SLOT, "slot")


def half_hour_to_time(h) -> pd.Timestamp | pd.DatetimeIndex:
    """Half-hour index h -> its start time."""
    return _index_to_time(h, SAMPLES_PER_HALF_HOUR)


def time_to_half_hour(t):
    """Time -> half-hour index h. t must fall exactly on a half-hour boundary."""
    return _time_to_index(t, SAMPLES_PER_HALF_HOUR, "half-hour")


@dataclass(frozen=True)
class BlockGrid:
    """A grid of blocks of rational length num/den samples, anchored at
    sample 0. Block b covers [b*num/den, (b+1)*num/den).
    """

    num: int
    den: int

    @property
    def length_samples(self) -> float:
        return self.num / self.den

    @staticmethod
    def floor(floor_level: int) -> "BlockGrid":
        """The floor grid for a given `mrd.floor_level` K: the 80-min detection
        window (240,000 samples = 1875 * 2^7) split into 2^K blocks.
        """
        return BlockGrid(1875, 2 ** (floor_level - 7))

    @staticmethod
    def integer(length_samples: int) -> "BlockGrid":
        """A grid whose blocks are a whole number of samples."""
        return BlockGrid(int(length_samples), 1)


BlockGrid.FINEST = BlockGrid(1875, 4)  # 9.375 s = 468.75 samples, fixed regardless of floor_level


def block_sums(values: np.ndarray, weights: np.ndarray, g0: int, grid: BlockGrid, b0: int, nb: int):
    """Sum `values` (weighted by `weights`, 0/1 usable) onto blocks b0..b0+nb-1
    of `grid`, splitting each sample's weight fractionally across the (at most
    two) blocks it straddles. `values` is (n,) or (n, m); the samples must
    fully cover the requested blocks. Returns (sums, weight_sums) with shapes
    (nb, m) or (nb,), and (nb,).
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    is_1d = values.ndim == 1
    x = values[:, None] if is_1d else values
    n, m = x.shape
    num, den = grid.num, grid.den

    assert g0 * den <= b0 * num, "samples start after block b0's start"
    assert (g0 + n) * den >= (b0 + nb) * num, "samples end before block (b0+nb)'s end"

    g = g0 + np.arange(n, dtype=np.int64)
    gden = g * den
    k = gden // num
    share1 = (np.minimum((k + 1) * num, (g + 1) * den) - gden) / den
    share2 = 1.0 - share1

    xw = np.where(weights[:, None] == 0, 0.0, x)  # zero x where weight is 0, so NaN never propagates

    sx = np.zeros((nb, m))
    sw = np.zeros(nb)
    for idx, share in ((k, share1), (k + 1, share2)):
        rel = idx - b0
        valid = (rel >= 0) & (rel < nb) & (share > 0)
        if not np.any(valid):
            continue
        sw += np.bincount(rel[valid], weights=(share * weights)[valid], minlength=nb)[:nb]
        flat_idx = rel[valid][:, None] * m + np.arange(m)[None, :]
        flat_w = (share[valid, None] * weights[valid, None]) * xw[valid]
        sx += np.bincount(flat_idx.ravel(), weights=flat_w.ravel(), minlength=nb * m)[: nb * m].reshape(nb, m)

    if is_1d:
        sx = sx[:, 0]
    return sx, sw


def majority_block(g, grid: BlockGrid):
    """Sample index g -> the block of `grid` containing the majority of it
    (for computations that need whole-sample block assignment).
    """
    g = np.asarray(g, dtype=np.int64) if not np.isscalar(g) else g
    return ((2 * g + 1) * grid.den) // (2 * grid.num)


def _reduced_grid(num: int, den: int) -> BlockGrid:
    g = gcd(num, den)
    return BlockGrid(num // g, den // g)


SLOT_MINUTE_CHOICES = (10, 15)


@dataclass(frozen=True)
class Cadence:
    """A run's time geometry: the output slot length S and the number of
    rungs on the tau ladder. Every window and block derives from these two:

    - slots tile each half-hour file (S divides 30 min) and are anchored at EPOCH;
    - the detection window is 8 S, centered on the slot: [start - 3.5 S, end + 3.5 S);
    - the ladder is 2 S / 2^j, j = 0..n_rungs-1 (ascending in `rung_seconds`):
      every rung up to S tiles the slot, and the top rung (2 S) is one window
      centered on the slot, [start - S/2, end + S/2);
    - the finest block is the smallest rung, 2^(n_rungs-2) of them per slot;
    - the MRD floor grid at level K splits the detection window into 2^K blocks.

    The default (10 min, 8 rungs: 9.375 s .. 20 min, 80-min detection window)
    is the original fixed design, and the module-level constants and
    BlockGrid.floor/FINEST above are that default. Pipeline code takes the
    run's own cadence (`Cadence.of(cfg)`), never the module-level defaults.
    """

    slot_min: int = 10
    n_rungs: int = 8

    def __post_init__(self):
        if self.slot_min not in SLOT_MINUTE_CHOICES:
            raise ValueError(f"slot_min must be one of {SLOT_MINUTE_CHOICES}, not {self.slot_min}")
        if self.n_rungs < 3:
            raise ValueError(f"n_rungs must be >= 3, not {self.n_rungs}")

    @classmethod
    def of(cls, cfg) -> "Cadence":
        """The cadence of a run Config (the default for a config-like object without one)."""
        section = getattr(cfg, "cadence", None)
        if section is None:
            return cls()
        return cls(slot_min=section.slot_min, n_rungs=section.n_rungs)

    # slots and files
    @property
    def samples_per_slot(self) -> int:
        return self.slot_min * 60 * SAMPLE_HZ

    @property
    def slot_s(self) -> float:
        return float(self.slot_min * 60)

    @property
    def slots_per_file(self) -> int:
        return SAMPLES_PER_HALF_HOUR // self.samples_per_slot

    def file_slots(self, h: int) -> list[int]:
        """The slots of half-hour file h, in order."""
        m = self.slots_per_file
        return list(range(m * h, m * h + m))

    def slot_file(self, k):
        """Slot index (or an array of them) -> the half-hour file holding it."""
        return k // self.slots_per_file

    def slot_to_time(self, k) -> pd.Timestamp | pd.DatetimeIndex:
        """Slot index k -> its start time."""
        return _index_to_time(k, self.samples_per_slot)

    def time_to_slot(self, t):
        """Time -> slot index k. t must fall exactly on a slot boundary."""
        return _time_to_index(t, self.samples_per_slot, "slot")

    # windows, in samples
    @property
    def detection_window_samples(self) -> int:
        return 8 * self.samples_per_slot

    @property
    def detection_margin_samples(self) -> int:
        return 7 * self.samples_per_slot // 2

    @property
    def detection_window_min(self) -> int:
        return 8 * self.slot_min

    def detection_window(self, k: int) -> tuple[int, int]:
        """(g0, n) of slot k's MRD detection window."""
        return self.samples_per_slot * k - self.detection_margin_samples, self.detection_window_samples

    @property
    def ladder_window_samples(self) -> int:
        return 2 * self.samples_per_slot

    @property
    def ladder_margin_samples(self) -> int:
        return self.samples_per_slot // 2

    def ladder_window(self, k: int) -> tuple[int, int]:
        """(g0, n) of slot k's top-rung window (the slot plus S/2 either side)."""
        return self.samples_per_slot * k - self.ladder_margin_samples, self.ladder_window_samples

    # the ladder
    @property
    def rung_seconds(self) -> tuple[float, ...]:
        """Every rung, ascending: 2 S / 2^(n_rungs-1), ..., S, 2 S."""
        top = 2 * self.slot_s
        return tuple(top / 2 ** (self.n_rungs - 1 - j) for j in range(self.n_rungs))

    @property
    def min_rung_s(self) -> float:
        return self.rung_seconds[0]

    @property
    def slot_rung_s(self) -> float:
        """The rung equal to the slot: the `naive` variant's fixed tau."""
        return self.slot_s

    @property
    def top_rung_s(self) -> float:
        """The one rung whose block is the 2 S window rather than tiling the slot."""
        return 2 * self.slot_s

    @property
    def finest_per_slot(self) -> int:
        return 2 ** (self.n_rungs - 2)

    @property
    def finest_per_window(self) -> int:
        return 2 * self.finest_per_slot

    @property
    def finest_per_half_hour(self) -> int:
        return self.finest_per_slot * self.slots_per_file

    @property
    def finest(self) -> BlockGrid:
        """The finest (smallest-rung) block grid."""
        return _reduced_grid(self.samples_per_slot, self.finest_per_slot)

    # MRD
    def floor(self, floor_level: int) -> BlockGrid:
        """The floor grid for `mrd.floor_level` K: the detection window split into 2^K blocks."""
        return _reduced_grid(self.detection_window_samples, 2 ** floor_level)

    def floor_blocks_per_half_hour(self, floor_level: int) -> int:
        return 2 ** floor_level * self.slots_per_file // 8

    def mode_scales_s(self, floor_level: int) -> np.ndarray:
        """P_i (seconds) for MRD mode i = 1..K: 2^i floor blocks."""
        floor_block_s = self.floor(floor_level).length_samples / SAMPLE_HZ
        return floor_block_s * 2.0 ** np.arange(1, floor_level + 1)


DEFAULT_CADENCE = Cadence()
