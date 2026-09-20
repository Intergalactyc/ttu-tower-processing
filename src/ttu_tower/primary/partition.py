"""Splitting the processing period into batches, around multi-file outages."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Batch:
    h_a: int
    h_b: int

    @property
    def id(self) -> str:
        return f"bat{self.h_a:07d}-{self.h_b:07d}"


def plan_batches(file_table: pd.DataFrame, period_slots: tuple[int, int], batch_max_files: int) -> tuple[list[Batch], list[int]]:
    """Split the half-hours overlapping the period's slots `[slot_a, slot_b)`
    into batches of at most `batch_max_files`, around outages (runs of >= 2
    consecutive non-accepted half-hours), which belong to no batch.
    """
    slot_a, slot_b = period_slots
    h_a, h_b = slot_a // 3, (slot_b - 1) // 3
    half_hours = list(range(h_a, h_b + 1))

    accepted = set(file_table.loc[file_table["status"] == "accepted", "half_hour"].tolist())
    is_accepted = [h in accepted for h in half_hours]

    outage_half_hours: list[int] = []
    in_batch = [True] * len(half_hours)
    i = 0
    while i < len(half_hours):
        if is_accepted[i]:
            i += 1
            continue
        j = i
        while j < len(half_hours) and not is_accepted[j]:
            j += 1
        if j - i >= 2:
            outage_half_hours.extend(half_hours[i:j])
            for idx in range(i, j):
                in_batch[idx] = False
        i = j

    batches: list[Batch] = []
    i = 0
    n = len(half_hours)
    while i < n:
        if not in_batch[i]:
            i += 1
            continue
        j = i
        while j < n and in_batch[j]:
            j += 1
        stretch = half_hours[i:j]
        for k in range(0, len(stretch), batch_max_files):
            chunk = stretch[k : k + batch_max_files]
            batches.append(Batch(h_a=chunk[0], h_b=chunk[-1]))
        i = j

    return batches, outage_half_hours
