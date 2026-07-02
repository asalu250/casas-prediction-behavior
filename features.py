"""
features.py
============
Turns the cleaned, occupancy-labeled event stream into one behavioral
feature vector per HOME day. AWAY days are excluded entirely from feature
building (see preprocessing.py for why) so the resulting matrix describes
in-home routine only.

Sensor taxonomy (informs feature design)
-----------------------------------------
This testbed's 17 usable sensors (MainDoor excluded, see preprocessing.py)
fall into three functional groups, not the generic "motion/door/item/light"
split assumed in early planning -- there are no light sensors in this
dataset, so "light usage" as originally scoped is not measurable and is
explicitly omitted rather than approximated with something misleading:

  MOTION  (PIR area sensors, mark presence in a room):
    DiningRoomAArea, KitchenAArea, LivingRoomAArea, BedroomAArea,
    BathroomAArea, HallwayA, MainEntryway, EntrywayB

  ITEM/INTERACTION (contact or pressure sensors on a specific object):
    KitchenADiningChair, LivingRoomAChair, KitchenASink, BedroomABed,
    KitchenAStove, BathroomASink, BathroomAToilet, KitchenARefrigerator

  DOOR (magnetic contact sensor on a door):
    BedroomADoor  (the only door sensor with meaningful event volume)

Every feature group below is built from this taxonomy.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

MOTION_SENSORS = [
    "DiningRoomAArea", "KitchenAArea", "LivingRoomAArea", "BedroomAArea",
    "BathroomAArea", "HallwayA", "MainEntryway", "EntrywayB",
]
ITEM_SENSORS = [
    "KitchenADiningChair", "LivingRoomAChair", "KitchenASink", "BedroomABed",
    "KitchenAStove", "BathroomASink", "BathroomAToilet", "KitchenARefrigerator",
]
DOOR_SENSORS = ["BedroomADoor"]

NIGHT_START_HOUR = 23   # 11pm
NIGHT_END_HOUR = 6      # 6am, exclusive
INACTIVITY_GAP_MINUTES = 60  # gap length that counts as a "long inactivity period" while awake


def _home_only(events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[~events["is_away"]].copy()


# ---------------------------------------------------------------------------
# 1. Hourly / time-of-day activity
# ---------------------------------------------------------------------------

def hourly_event_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """
    Rows = date, columns = hour 0-23, values = total sensor events in
    that hour. This is the base matrix several other features derive
    from (time-of-day activity, sleep activity, routine consistency).
    """
    home = _home_only(events)
    home["hour"] = home["datetime"].dt.hour
    mat = (
        home.groupby(["date", "hour"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    return mat


def time_of_day_distribution(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of each day's activity falling in morning / afternoon /
    evening / night. Using *fractions* rather than raw counts matters:
    a generally more or less active day shouldn't be conflated with a
    day whose activity has genuinely shifted timing -- normalizing by
    the day's own total isolates the *timing* signal from the *volume*
    signal, which get modeled as separate features on purpose.
    """
    bins = {
        "morning_frac": range(6, 12),
        "afternoon_frac": range(12, 18),
        "evening_frac": range(18, 23),
        "night_frac": list(range(23, 24)) + list(range(0, 6)),
    }
    total = hourly.sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=hourly.index)
    for name, hours in bins.items():
        cols = [h for h in hours if h in hourly.columns]
        out[name] = hourly[cols].sum(axis=1) / total
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Daily totals & per-sensor-group activity
# ---------------------------------------------------------------------------

def daily_group_activity(events: pd.DataFrame) -> pd.DataFrame:
    home = _home_only(events)
    out = pd.DataFrame(index=sorted(home["date"].unique()))
    out.index.name = "date"

    def count_group(sensors, colname):
        sub = home[home["sensor"].isin(sensors)]
        counts = sub.groupby("date").size()
        out[colname] = counts

    count_group(MOTION_SENSORS, "motion_events")
    count_group(ITEM_SENSORS, "item_events")
    count_group(DOOR_SENSORS, "door_events")
    out["total_events"] = home.groupby("date").size()
    out["unique_sensors_active"] = home.groupby("date")["sensor"].nunique()
    return out.fillna(0)


# ---------------------------------------------------------------------------
# 3. Sleep / nighttime activity
# ---------------------------------------------------------------------------

def sleep_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates sleep-related behavior from the BedroomABed contact
    sensor and overall nighttime (11pm-6am) motion. We can't observe
    sleep directly (no dedicated sleep sensor), so this is a behavioral
    proxy, not a clinical sleep measurement -- worth stating plainly in
    the report's limitations section.

    - bed_night_events: BedroomABed ON/OFF transitions during the night
      window (restlessness/getting up proxy -- more transitions can mean
      more disrupted sleep, though a single bug in the sensor debounce
      could also inflate this, which is a caveat worth naming).
    - night_motion_events: any motion sensor firing 11pm-6am (getting up,
      bathroom trips, etc.)
    - first_night_activity_hour / last_night_activity_hour: approximate
      bedtime / wake time signal.
    """
    home = _home_only(events)
    home["hour"] = home["datetime"].dt.hour
    is_night = (home["hour"] >= NIGHT_START_HOUR) | (home["hour"] < NIGHT_END_HOUR)
    night = home[is_night].copy()

    # night events logged after midnight belong to the PREVIOUS night's
    # "sleep date" (e.g. 1am Tuesday is still Monday night's sleep)
    night["sleep_date"] = night["date"]
    after_midnight = night["hour"] < NIGHT_END_HOUR
    night.loc[after_midnight, "sleep_date"] = (
        pd.to_datetime(night.loc[after_midnight, "date"]) - pd.Timedelta(days=1)
    ).dt.date

    out = pd.DataFrame(index=sorted(home["date"].unique()))
    out.index.name = "date"

    bed_night = night[night["sensor"].isin(["BedroomABed"])]
    out["bed_night_events"] = bed_night.groupby("sleep_date").size()

    motion_night = night[night["sensor"].isin(MOTION_SENSORS)]
    out["night_motion_events"] = motion_night.groupby("sleep_date").size()

    return out.fillna(0)


# ---------------------------------------------------------------------------
# 4. Long inactivity periods
# ---------------------------------------------------------------------------

def inactivity_features(events: pd.DataFrame,
                         gap_minutes: int = INACTIVITY_GAP_MINUTES) -> pd.DataFrame:
    """
    Counts gaps of >= gap_minutes between consecutive sensor events
    within a day (restricted to waking hours, 6am-11pm, so normal
    overnight sleep gaps aren't double-counted as "inactivity"). A rising
    trend in daytime inactivity gaps is one of the more literature-
    supported early indicators of behavioral/functional decline in
    smart-home aging-in-place research, which is exactly the kind of
    signal this project is trying to surface -- so this feature is one
    of the more important ones for the eventual "predict future
    deviation" objective, not just a generic engineering exercise.
    """
    home = _home_only(events).sort_values("datetime")
    home["hour"] = home["datetime"].dt.hour
    waking = home[(home["hour"] >= 6) & (home["hour"] < 23)].copy()

    waking["gap_min"] = waking.groupby("date")["datetime"].diff().dt.total_seconds() / 60

    out = pd.DataFrame(index=sorted(home["date"].unique()))
    out.index.name = "date"
    long_gaps = waking[waking["gap_min"] >= gap_minutes]
    out["n_long_inactivity_gaps"] = long_gaps.groupby("date").size()
    out["max_inactivity_gap_min"] = waking.groupby("date")["gap_min"].max()
    return out.fillna(0)


# ---------------------------------------------------------------------------
# 5. Room transitions
# ---------------------------------------------------------------------------

def room_transition_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Counts how often consecutive motion events occur in *different*
    rooms (a proxy for physical mobility/movement around the home) vs.
    the same room repeated. Only ON events from motion sensors are used,
    since ON marks entry into a room's detection zone.
    """
    home = _home_only(events)
    motion_on = home[(home["sensor"].isin(MOTION_SENSORS)) & (home["message"] == "ON")]
    motion_on = motion_on.sort_values("datetime")

    out = pd.DataFrame(index=sorted(home["date"].unique()))
    out.index.name = "date"

    def transitions_per_day(g):
        rooms = g["sensor"].values
        if len(rooms) < 2:
            return 0
        return int((rooms[1:] != rooms[:-1]).sum())

    out["room_transitions"] = motion_on.groupby("date").apply(
        transitions_per_day, include_groups=False
    )
    return out.fillna(0)


# ---------------------------------------------------------------------------
# 6. Routine consistency
# ---------------------------------------------------------------------------

def routine_consistency_features(hourly: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Measures how similar each day's hourly activity *shape* is to the
    person's own recent typical shape, using cosine similarity between
    the day's 24-dim hourly profile (normalized to sum to 1, so this is
    about timing/pattern, not volume) and the mean profile of the prior
    `window` HOME days. Lower similarity = the day's rhythm deviated
    from the person's own recent routine.

    Cosine similarity is used over Euclidean distance here because we
    specifically want to compare *shape* (when during the day activity
    happens) independent of overall activity level, which cosine
    similarity naturally provides and Euclidean distance does not
    (Euclidean would conflate "same rhythm, less active" with "different
    rhythm entirely").
    """
    profiles = hourly.div(hourly.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    out = pd.DataFrame(index=hourly.index)
    sims = []
    for i, date in enumerate(profiles.index):
        prior = profiles.iloc[max(0, i - window):i]
        if len(prior) == 0:
            sims.append(np.nan)
            continue
        ref = prior.mean(axis=0).values
        today = profiles.loc[date].values
        denom = (np.linalg.norm(ref) * np.linalg.norm(today))
        sims.append(float(np.dot(ref, today) / denom) if denom > 0 else np.nan)
    out["routine_similarity_cos"] = sims
    return out


# ---------------------------------------------------------------------------
# 7. Rolling statistics (applied last, over the assembled feature matrix)
# ---------------------------------------------------------------------------

def add_rolling_stats(feat: pd.DataFrame, columns: list[str],
                       windows: tuple[int, ...] = (7, 14)) -> pd.DataFrame:
    """
    Adds rolling mean & std for the given columns. These let downstream
    models see whether *today* is unusual relative to the person's own
    recent baseline (short window) and their longer-term norm (longer
    window), which is closer to how a clinician or caregiver would
    actually judge "is this a change" than looking at any single day in
    isolation.
    """
    feat = feat.copy()
    for col in columns:
        for w in windows:
            feat[f"{col}_roll{w}_mean"] = feat[col].rolling(w, min_periods=max(2, w // 2)).mean()
            feat[f"{col}_roll{w}_std"] = feat[col].rolling(w, min_periods=max(2, w // 2)).std()
    return feat


# ---------------------------------------------------------------------------
# 8. Day-of-week / calendar features
# ---------------------------------------------------------------------------

def calendar_features(index: pd.Index) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(index))
    out = pd.DataFrame(index=index)
    out["day_of_week"] = dates.dt.dayofweek.values
    out["is_weekend"] = (dates.dt.dayofweek >= 5).values.astype(int)
    return out


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_feature_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles the full daily feature matrix from cleaned, occupancy-
    labeled events. One row per HOME day.
    """
    hourly = hourly_event_matrix(events)
    tod = time_of_day_distribution(hourly)
    daily = daily_group_activity(events)
    sleep = sleep_features(events)
    inact = inactivity_features(events)
    trans = room_transition_features(events)
    routine = routine_consistency_features(hourly)

    feat = daily.join([tod, sleep, inact, trans, routine], how="left")
    feat = feat.join(calendar_features(feat.index))

    rolling_cols = [
        "total_events", "motion_events", "item_events", "door_events",
        "n_long_inactivity_gaps", "night_motion_events", "room_transitions",
    ]
    feat = add_rolling_stats(feat, rolling_cols)

    feat.index = pd.to_datetime(feat.index)
    feat = feat.sort_index()
    return feat


if __name__ == "__main__":
    events = pd.read_parquet("data/events_clean.parquet")
    feat = build_feature_matrix(events)
    feat.to_parquet("data/features_daily.parquet")
    print("Feature matrix:", feat.shape)
    print(feat.columns.tolist())
