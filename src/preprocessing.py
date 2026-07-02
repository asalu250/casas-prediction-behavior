"""
preprocessing.py
=================
Loads the raw CASAS-format smart-home event log and produces a cleaned,
chronologically-sorted event stream with an "occupancy state" label
(HOME / AWAY) attached to every day.

Why an occupancy state matters
-------------------------------
This dataset covers ~369 days of one occupant's home. Exploratory analysis
showed several multi-day stretches with zero or near-zero sensor activity
(e.g. Aug 1-12, Dec 24-Jan 14). These are not sensor failures — they are
almost certainly periods where the occupant was physically absent
(travel, extended stay elsewhere).

If we feed raw daily activity into a behavioral-deviation model without
accounting for this, every absence will register as a massive "anomaly",
drowning out the subtler in-home routine drift the project actually cares
about (the CASAS literature on this exact problem: e.g. Cook et al. on
smart-home activity recognition explicitly separate occupancy detection
from activity/routine modeling for this reason).

So preprocessing here does two things:
1. Standard cleaning: parse timestamps (handling the rare rows missing a
   microsecond fraction), sort chronologically, drop the OPEN/CLOSE
   MainDoor stream (39 events total, functionally unused sensor -- an
   almost-certainly broken or disconnected door sensor, not a reliable
   signal) into a separate table so it doesn't get silently mixed with
   ON/OFF motion-style sensors.
2. Occupancy labeling: classify each calendar day as HOME or AWAY using an
   explicit, reproducible rule (see `label_occupancy`), so that downstream
   feature engineering and modeling can restrict the "normal behavior"
   baseline to HOME days only.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

RAW_COLUMNS = ["datetime_str", "sensor", "message"]


def load_raw_events(path: str | Path) -> pd.DataFrame:
    """
    Load the raw CASAS-format tab-separated event log.

    Format per line: <date> <time>\t<sensor>\t<message>\t(trailing tab)
    Timestamps are almost always fractional-second precision, but a
    handful (7 out of 3.7M in this dataset) omit the microsecond
    fraction -- format='mixed' handles both without dropping rows.
    """
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=RAW_COLUMNS + ["_blank"], usecols=[0, 1, 2],
        engine="c",
    )
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="mixed")
    df = df.drop(columns=["datetime_str"])
    df = df.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    return df[["datetime", "sensor", "message"]]


def split_door_and_motion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    MainDoor logs OPEN/CLOSE (only 39 events total in this dataset) while
    every other sensor logs ON/OFF. Mixing binary vocabularies would make
    later one-hot/state features ambiguous, and MainDoor's near-total
    silence over a full year (should fire far more often for an entry
    door) suggests it's disconnected or malfunctioning for most of the
    study period, so it is not a reliable feature source. We keep it
    separately in case it's useful for a sanity check, but exclude it
    from the main motion/usage event stream used for feature engineering.
    """
    is_door = df["sensor"] == "MainDoor"
    door = df[is_door].copy()
    motion = df[~is_door].copy()
    return motion, door


def clean_events(raw_path: str | Path) -> pd.DataFrame:
    """Full cleaning pipeline: load -> sort -> drop unreliable sensor."""
    df = load_raw_events(raw_path)
    motion, _door = split_door_and_motion(df)
    # message should be strictly ON/OFF at this point
    assert set(motion["message"].unique()) <= {"ON", "OFF"}
    motion["date"] = motion["datetime"].dt.date
    return motion.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Occupancy (HOME / AWAY) labeling
# ---------------------------------------------------------------------------

def daily_event_counts(df: pd.DataFrame) -> pd.Series:
    """Total sensor events per calendar day, reindexed to include days
    with zero events (which would otherwise be silently missing)."""
    counts = df.set_index("datetime").resample("D").size()
    return counts


def label_occupancy(
    df: pd.DataFrame,
    zero_day_threshold: int = 15,
    min_away_run: int = 2,
) -> pd.DataFrame:
    """
    Classify each calendar day as HOME or AWAY.

    Rule (deliberately simple & auditable rather than a black-box model,
    since this label feeds directly into what counts as "normal" for
    every downstream model):

      1. A day is a *candidate* AWAY day if its total event count is
         below `zero_day_threshold`. A genuinely occupied home triggers
         motion/light/appliance sensors far more than that even on a
         quiet day (the empirical 5th-percentile floor for normal days
         in this dataset is ~80+ events/day; 15 gives comfortable margin
         without being so strict that a single quiet Sunday gets
         mislabeled).
      2. A candidate day only becomes a *confirmed* AWAY day if it's part
         of a run of at least `min_away_run` consecutive candidate days.
         This avoids mislabeling isolated low-activity days (e.g. a
         single sick day spent mostly in bed, which is exactly the kind
         of legitimate behavioral signal we don't want to throw away) as
         travel absences, while still catching genuine multi-day trips.

    Returns a DataFrame indexed by date with columns:
      event_count, is_candidate_away, is_away (final label)
    """
    counts = daily_event_counts(df)
    full_index = pd.date_range(counts.index.min().normalize(),
                                counts.index.max().normalize(), freq="D")
    counts = counts.reindex(full_index, fill_value=0)

    occ = pd.DataFrame({"event_count": counts})
    occ["is_candidate_away"] = occ["event_count"] < zero_day_threshold

    # find runs of consecutive candidate-away days
    run_id = (occ["is_candidate_away"] != occ["is_candidate_away"].shift()).cumsum()
    run_lengths = occ.groupby(run_id)["is_candidate_away"].transform("size")
    occ["is_away"] = occ["is_candidate_away"] & (run_lengths >= min_away_run)

    occ.index.name = "date"
    return occ


def attach_occupancy(df: pd.DataFrame, occupancy: pd.DataFrame) -> pd.DataFrame:
    """Merge the per-day HOME/AWAY label onto the event-level dataframe."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    merged = df.merge(
        occupancy[["is_away"]].reset_index().rename(columns={"date": "date"}),
        on="date", how="left",
    )
    merged["is_away"] = merged["is_away"].fillna(False)
    return merged


# ---------------------------------------------------------------------------
# Convenience top-level entry point
# ---------------------------------------------------------------------------

def run_preprocessing(raw_path: str | Path):
    """
    Returns:
        events: event-level dataframe (datetime, sensor, message, date, is_away)
        occupancy: day-level dataframe (event_count, is_candidate_away, is_away)
    """
    events = clean_events(raw_path)
    occupancy = label_occupancy(events)
    events = attach_occupancy(events, occupancy)
    return events, occupancy


if __name__ == "__main__":
    import sys
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/ucd002.txt"
    events, occupancy = run_preprocessing(raw_path)
    print(f"Loaded {len(events):,} events across {events['date'].nunique()} days")
    print(f"AWAY days: {occupancy['is_away'].sum()} / {len(occupancy)}")
    events.to_parquet("data/events_clean.parquet")
    occupancy.to_parquet("data/occupancy_daily.parquet")
