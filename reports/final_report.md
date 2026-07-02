# Predicting Behavioral Deviation in Smart-Home Sensor Data
### A Trend-Based Approach on the CASAS `ucd002` Testbed

---

## 1. Motivation & Research Question

Smart-home sensor networks passively record an occupant's daily routine at fine temporal resolution.
Most published work on this kind of data focuses on **activity recognition** (labeling what someone is
doing right now) or **post-hoc anomaly detection** (flagging a day as unusual after it happens). This
project asks a different question:

> Can trends in day-to-day behavioral features be used to anticipate a deviation from an occupant's
> established routine, rather than only detecting it retrospectively?

This framing is motivated by applications such as monitoring older adults living independently, where
a *gradual drift* in routine — increasing inactivity, disrupted sleep, reduced household mobility — is
often a more clinically meaningful and earlier signal than any single dramatic "anomalous" day.

## 2. Dataset

The CASAS `ucd002` testbed provides approximately one year (2025-04-03 to 2026-04-07, 369 days) of raw
sensor events from one occupant's home: 3,703,825 events across 18 sensors, logged as
`datetime | sensor | message`.

**Sensor taxonomy** (determined empirically, not assumed from the original project brief):

| Group | Sensors | Notes |
|---|---|---|
| Motion (PIR, room presence) | DiningRoomAArea, KitchenAArea, LivingRoomAArea, BedroomAArea, BathroomAArea, HallwayA, MainEntryway, EntrywayB | ON/OFF |
| Item / interaction | KitchenADiningChair, LivingRoomAChair, KitchenASink, BedroomABed, KitchenAStove, BathroomASink, BathroomAToilet, KitchenARefrigerator | ON/OFF |
| Door | BedroomADoor | ON/OFF |
| Door (unreliable) | MainDoor | OPEN/CLOSE, only 39 events in the entire year — excluded |

There are **no light sensors** in this testbed. The originally planned "light usage" feature is
therefore not measurable and is omitted rather than approximated with a misleading proxy — this is
stated here explicitly as a scope limitation rather than left implicit.

## 3. Data Quality Findings and Their Methodological Consequences

Initial inspection surfaced three issues, the first of which materially changed the project design:

1. **Extended zero/near-zero activity periods**: Aug 1–12 (12 days), Dec 24–Jan 14 (22 days), and a
   ragged low-activity stretch in mid-October. These read as occupant-absence (travel) periods, not
   sensor failure.
2. A single out-of-order timestamp (one sensor debounce artifact, not a systemic issue).
3. 7 of 3.7M timestamps missing the microsecond fraction (parsed transparently, no data lost).

**Why the absence periods required a design change:** the research question is about deviation from
*this person's own established in-home routine*. If raw daily activity is fed directly into a
deviation/anomaly model, every travel period registers as an extreme anomaly, which is true but
uninteresting — it drowns out the much subtler in-home routine drift the project actually cares about.

**Response:** an explicit, auditable **occupancy state (HOME / AWAY)** is computed before any
behavioral feature is built. A day is a *candidate* AWAY day if its total event count falls below an
empirically grounded threshold, and becomes a *confirmed* AWAY day only if part of a run of 2+
consecutive candidate days — the run requirement is a deliberate choice to avoid misclassifying a
single genuinely low-activity day (e.g., a day spent resting, which is itself meaningful signal) as
travel. This rule correctly recovers all three known absence periods.

Of 370 calendar days, **320 are classified HOME** and used for behavioral modeling; **50 are AWAY**
and excluded. One known edge case (Oct 28, 2 events, isolated rather than part of a run) is not
flagged AWAY and is documented as a limitation of the rule rather than hidden.

## 4. Feature Engineering

45 features are computed per HOME day, spanning:

- **Activity volume**: total events, per-sensor-group counts (motion / item / door), unique sensors active
- **Time-of-day distribution**: fraction of activity in morning / afternoon / evening / night —
  normalized by the day's own total, so timing shifts are measured independently of overall activity
  level
- **Sleep-proxy features**: nighttime (11pm–6am) bed-sensor and motion events, attributed to the
  correct "sleep date" (post-midnight events belong to the previous night)
- **Long daytime inactivity gaps**: count and max duration of gaps ≥ 60 minutes during waking hours —
  a feature with direct support in prior aging-in-place smart-home literature as an early indicator of
  functional decline
- **Room transitions**: count of consecutive motion events occurring in different rooms, as a mobility
  proxy
- **Routine similarity**: cosine similarity between a day's normalized hourly activity shape and the
  mean shape of the prior 7 HOME days — cosine similarity is used specifically because it isolates
  *rhythm* from *volume*, which Euclidean distance would conflate
- **Calendar**: day of week, weekend flag
- **Rolling statistics**: 7- and 14-day rolling mean/std for the core activity counts, so models can
  compare "today" against the person's own recent and longer-term baseline

## 5. Label Generation (No Ground Truth Available)

This dataset has no clinician-verified "behavioral change" labels. Rather than pick one labeling
heuristic and treat it as ground truth, two independent method families were used so their agreement
could be checked:

1. **Statistical deviation label (primary):** a day is "deviant" if ≥3 features simultaneously exceed
   a rolling 14-day z-score threshold of 2.0 (rolling stats computed on the *trailing* window,
   excluding the day itself, so a day can't inflate its own baseline). Requiring multiple co-occurring
   deviations reduces false positives from a single noisy feature.
2. **Unsupervised anomaly scores (validation):** Isolation Forest and Local Outlier Factor, trained
   with no label at all, purely on the standardized feature matrix.

**Result:** the statistical label flags 16.2% of HOME days as deviant. Agreement with the label-blind
methods (Cohen's kappa) is fair-to-moderate — roughly 0.4 against Isolation Forest and 0.24 against
LOF. This indicates the statistical label captures a real, structurally detectable pattern (not pure
noise from the labeling rule), while also confirming the methods are not redundant with each other.
This agreement check is treated as evidence *for*, not proof *of*, the label's validity — an important
distinction given no ground truth exists to fully validate against.

## 6. Modeling

**Supervised:** Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost — trained against
the statistical label.

**Unsupervised:** Isolation Forest, One-Class SVM, Local Outlier Factor — trained with no label,
compared post-hoc against the statistical label as an independent structural check.

**Chronological evaluation, not random splitting.** Because the project is framed around *prediction*,
a random shuffled train/test split would let a model see information from after the day it's
predicting — which is not a prediction task at all. All train/test splits and cross-validation
(`TimeSeriesSplit`) are strictly forward-in-time.

**Leakage control:** the same-day z-score columns used to construct the label are excluded from the
feature set given to models, so a model cannot trivially "predict" the label by reading back the exact
values used to generate it.

### Results (holdout, chronological split)

| Model | F1 | Precision | Recall | ROC-AUC | CV F1 (mean ± std) |
|---|---|---|---|---|---|
| Logistic Regression | 0.33 | 0.38 | 0.30 | 0.73 | 0.35 ± 0.15 |
| Random Forest | 0.57 | 1.00 | 0.40 | 0.84 | 0.17 ± 0.23 |
| Gradient Boosting | 0.56 | 0.63 | 0.50 | 0.79 | 0.30 ± 0.11 |
| SVM | 0.67 | 0.64 | 0.70 | 0.88 | 0.42 ± 0.17 |
| XGBoost | 0.50 | 0.67 | 0.40 | 0.72 | 0.41 ± 0.10 |

SVM performed best on the single holdout split; XGBoost and SVM had the most stable
cross-validated F1. Random Forest's very high holdout precision (1.00) paired with much lower and
more volatile CV performance is a sign of overfitting to the specific holdout window rather than a
robustly better model — a good example of why a single holdout split should never be trusted alone,
and cross-validation is reported alongside it.

Unsupervised methods (never shown the label) recovered a meaningful fraction of the same days
(F1 ≈ 0.24–0.27 against the statistical label without training on it), supporting — though not
proving — that flagged days are structurally distinguishable, not an artifact of the labeling rule.

### Feature Importance

Using **permutation importance** (comparable across model types, unlike each model's native importance
measure) on the best-performing model, the top features were: `routine_similarity_cos`,
`total_events`, `motion_events`, `night_frac`, `room_transitions`, and `item_events` — i.e., both
*how much* the person moved through the home and *whether the day's rhythm resembled their own recent
pattern* were the strongest predictors, more so than raw sleep-proxy counts alone.

## 7. Limitations

- **No ground-truth labels.** All reported metrics measure agreement with a self-constructed
  statistical heuristic, not validated behavioral-change events. Precision/recall/F1 should be read
  as "agreement with our deviation definition," not "accuracy at detecting real health changes."
- **Small effective sample.** ~314 usable days, ~50 positive examples after dropping rolling-window
  warmup rows — cross-validation variance is consequently high, visible directly in the CV std columns
  above.
- **Single occupant, single home.** No claim of generalization to other households, sensor layouts, or
  living situations.
- **No light sensors available** in this testbed; the originally planned "light usage" feature is not
  present.
- **Sleep is inferred, not measured** — bed-sensor and nighttime-motion features are behavioral
  proxies, not clinical sleep signal.
- **Occupancy detection edge case**: an isolated 2-event day (Oct 28) is not flagged AWAY under the
  run-length rule, illustrating that the rule, while principled and auditable, is not perfect at every
  boundary.

## 8. Future Work

- Longitudinal validation against actual outcome data (even self-reported "how are you feeling"
  entries) to test whether statistically-flagged days correlate with anything the occupant would
  independently call meaningful
- Sequence models (LSTM/GRU) directly on the hourly event matrix, to test whether they discover
  structure the hand-engineered daily features miss
- Multi-home validation of the feature set and thresholds
- Explicit N-day-ahead prediction framing, using only information available strictly before the
  prediction horizon, rather than same-day classification
- Calibrated probabilities and a cost-sensitive decision threshold — in a real deployment, missed
  genuine changes and false alarms carry very different costs, which a simple F1-optimized threshold
  does not reflect

## 9. Reproducing This Analysis

See `README.md` in the project root for setup instructions. The full, executed analysis is in
`notebooks/main_analysis.ipynb`; reusable pipeline code lives in `src/`.
