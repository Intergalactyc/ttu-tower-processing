"""Shared constants and helpers for the viewer tests' synthetic runs."""
from ttu_tower.timegrid import half_hour_to_time

# primary-only run: 1 boom, 6 core half-hours in 2 batches, one missing file
PRIMARY_HALF_HOURS = list(range(7000, 7008))
PRIMARY_MISSING = [7004]
FAULT_HALF_HOUR = 7002
BOUNDS_INDEX = 30_000  # u far outside the sonic bounds
W_SPIKE_INDEX = 45_000  # a lone w spike: ue/vn lose the sample only by coupling
TS_SPIKE_INDEX = 60_000

FULL_HALF_HOURS = list(range(6100, 6108))
FULL_BOOMS = [1, 2]
SPIKY_HALF_HOUR, SPIKY_BOOM = 6105, 2
SPIKY_SLOT = 3 * SPIKY_HALF_HOUR + 1  # w spikes every 50 samples: its spike fraction fails tertiary's filter


def mutate_primary(h, boom, df):
    if h != FAULT_HALF_HOUR:
        return df
    df = df.copy()
    df.loc[df.index[BOUNDS_INDEX], f"u_{boom}"] = 500.0
    df.loc[df.index[W_SPIKE_INDEX], f"w_{boom}"] += 12.0
    df.loc[df.index[TS_SPIKE_INDEX], f"ts_{boom}"] += 25.0
    return df


def mutate_full(h, boom, df):
    if (h, boom) != (SPIKY_HALF_HOUR, SPIKY_BOOM):
        return df
    df = df.copy()
    rows = df.index[30_000:60_000:50]
    df.loc[rows, f"w_{boom}"] += 12.0
    return df


def write_config(path, tag, raw_dir, half_hours, booms, batch_max_files):
    start = half_hour_to_time(half_hours[0] + 1).strftime("%Y-%m-%d %H:%M")
    end = half_hour_to_time(half_hours[-1]).strftime("%Y-%m-%d %H:%M")
    raw = str(raw_dir).replace("\\", "/")
    booms_toml = ", ".join(str(b) for b in booms)
    path.write_text(
        f'tag = "{tag}"\n'
        f'[paths]\nraw_dirs = ["{raw}"]\n'
        f'[period]\nstart = "{start}"\nend = "{end}"\n'
        f"[files]\nbad_records = []\nbooms = [{booms_toml}]\n"
        f"[primary]\nbatch_max_files = {batch_max_files}\n"
        f"[tertiary]\nveer_reference_boom = {booms[0]}\n"
        f"[tertiary.fits]\nalpha_booms = [{booms_toml}]\ngamma_booms = [{booms_toml}]\n"
        f"wdgamma_booms = [{booms_toml}]\nloglaw_booms = [{booms_toml}]\n"
    )
    return path
