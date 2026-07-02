# CASAS Smart-Home Behavioral Deviation Prediction

A machine learning research pipeline that learns an occupant's normal behavioral patterns from
smart-home sensor data (CASAS `ucd002` testbed) and evaluates whether trends in daily behavior can
anticipate deviations from that routine — a building block for applications like aging-in-place
health monitoring.

**Start here:** [`reports/final_report.md`](reports/final_report.md) for the full methodology,
results, and discussion. [`notebooks/main_analysis.ipynb`](notebooks/main_analysis.ipynb) for the
complete, executed, documented walkthrough.

## Project structure

```
casas-behavior-prediction/
├── data/                    # raw & processed data (NOT committed — see below)
├── src/
│   ├── preprocessing.py     # load, clean, occupancy (HOME/AWAY) detection
│   ├── features.py          # 45-feature daily behavioral feature engineering
│   ├── labeling.py          # statistical + unsupervised deviation labels (no ground truth exists)
│   ├── models.py             # supervised + unsupervised model training & comparison
│   └── evaluate.py          # feature importance & visualization utilities
├── notebooks/
│   └── main_analysis.ipynb  # full documented walkthrough, already executed
├── reports/
│   └── final_report.md      # methodology, results, limitations, future work
├── figures/                 # generated plots (created by running the pipeline)
├── requirements.txt
└── README.md
```

## About the data

The raw file (`ucd002.txt`, ~165MB, 3.7M rows) is **not included in this repo** — GitHub blocks
files over 100MB, and committing large raw datasets isn't good practice regardless. To reproduce:

1. Place your copy of the CASAS `ucd002` event log at `data/ucd002.txt`
2. Run the pipeline (see below) — it regenerates everything else in `data/` as `.parquet` files,
   which are also excluded from git via `.gitignore`

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

Either run the notebook (recommended — it's the documented walkthrough):

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

...or run the modules directly, in order, from the project root:

```bash
python src/preprocessing.py data/ucd002.txt   # -> data/events_clean.parquet, data/occupancy_daily.parquet
python src/features.py                         # -> data/features_daily.parquet
python src/labeling.py                          # -> data/labels_daily.parquet
python src/models.py                            # prints model comparison to stdout
```

## Key methodological decisions (see final report for full detail)

- **Occupancy detection**: extended zero-activity periods in the raw data are travel/absence, not
  sensor failure. These are detected and excluded from "normal routine" modeling *before* any
  behavioral feature is built, so vacations don't get mistaken for health-related anomalies.
- **No ground-truth labels exist** for this dataset, so labels are generated two independent ways
  (statistical z-score deviation + unsupervised anomaly detection) and their agreement is reported,
  rather than trusting one heuristic blindly.
- **Chronological (not random) train/test splitting**, since the project is framed around prediction —
  a model must only ever see the past when evaluated on the future.

## Requirements

Python 3.10+, see `requirements.txt` for package versions.
