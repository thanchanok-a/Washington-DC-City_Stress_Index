# Washington-DC-City_Stress_Index
Fuses **weather**, **news sentiment**, and **transit delays** into a single 0–100 score that measures — and forecasts — a city's day-to-day tension.

Most data projects lean on a single source. This one fuses three live signals into one composite metric, stores them in SQLite, and trains a time-series model to forecast tomorrow's stress level.

## What It Does

| Signal | Source | What It Measures |
|---|---|---|
| Weather | Open-Meteo API (free, no key) | Heat, rain, and wind pressure on mood |
| News Sentiment | NewsAPI + VADER NLP | Negativity in local headlines |
| Transit Stress | WMATA API (rail incidents + bus positions) | Delay rates and overcrowding |

These are combined into a **City Stress Index (CSI, 0–100)**, then a Gradient Boosting model forecasts the next day's score.

## Tech Stack

`requests` · `sqlite3` · `pandas` · `numpy` · `vaderSentiment` · `scikit-learn` · `plotly` · `matplotlib` / `seaborn`

## Latest Run Results

| Metric | This run | Previous run |
|---|---|---|
| Mean CSI (30-day window) | 39.3 / 100 | 40.2 / 100 |
| Peak CSI | 92.0 on 2026-07-21 (Critical — overcrowding incident) | 92.0 (same day) |
| Calm days | 8 | 8 |
| High + Critical days | 5 | 6 |
| Cross-validated MAE | **13.11 ± 5.52** CSI points | 14.14 ± 4.75 |
| Final in-sample MAE | 9.09 CSI points | 9.35 |
| Tomorrow's forecast | 46.0 / 100 — Elevated | 47.1 / 100 — Elevated |
| Rain stress ↔ delay rate correlation | +0.023 (essentially none) | +0.023 |
| Wind stress ↔ delay rate correlation | +0.148 (weak) | +0.148 |

**Highest-stress days:** Tuesday (avg 54.3) and Monday (avg 47.6) — consistent with commuter-driven friction dominating the index. **Calmest:** weekends (Sat 18.9, Sun 16.6).

**Feature importance:** `csi_lag_1` (yesterday's CSI) and `csi_roll3_mean` (3-day rolling average) are the strongest predictors, followed by `raw_transit`. `raw_sentiment` contributed almost nothing this run — see Known Issues below for why.

## Pipeline

1. **Configuration** — city coordinates, timezone, and baseline transit delay rate set via a single config dict.
2. **Weather ingestion** — pulls 30 days of daily temperature, precipitation, and wind from Open-Meteo, then derives `weather_stress`, `heat_stress`, `rain_stress`, and `wind_stress` sub-scores.
3. **News sentiment** — fetches recent headlines via NewsAPI, filters for city relevance, and scores each with VADER.
4. **Transit stress** — combines live WMATA rail incident severity and bus schedule deviation (today) with a calibrated historical simulation (prior days).
5. **SQLite fusion** — all three signals are loaded into `city_stress.db` and joined via SQL into a single fused table.
6. **CSI calculation** — a weighted composite (Weather 30% / News 35% / Transit 35%), scaled to 8–92, bucketed into **Calm / Elevated / High / Critical** risk tiers.
7. **SQL insights** — queries surface stress by day of week, by weather condition, the worst 5 days, and stress during transit incidents.
8. **Forecasting** — a compact `GradientBoostingRegressor` (6 engineered features, `TimeSeriesSplit` cross-validation) predicts tomorrow's CSI.
9. **Dashboard** — a 6-panel interactive Plotly dashboard exported to HTML.

## Output Files

```
outputs/
├── city_stress_dashboard.html   # interactive 6-panel Plotly dashboard
└── feature_importance.png       # model feature importance chart
city_stress.db                   # SQLite database (weather, news, transit, city_stress_index tables)
```

## Setup

```bash
pip install vaderSentiment pandas numpy scikit-learn plotly matplotlib seaborn requests
```

You'll need free API keys from:
- [NewsAPI](https://newsapi.org) — for news headlines
- [WMATA Developer Portal](https://developer.wmata.com) — for live rail/bus data

**Store them as Colab Secrets (or environment variables) — never hardcode them, and never
put a bare `userdata.get('KEY_NAME')` expression as the last line of a cell**, since Colab
auto-displays the return value as output. That prints the literal secret into the saved
notebook file even if the code itself never hardcodes it. Assign it to a variable and print
nothing, e.g.:

```python
from google.colab import userdata
NEWS_API_KEY = userdata.get('NEWS_API_KEY')  # do not leave this as the last line alone
```

Open-Meteo requires no key.

> ⚠️ **If you're publishing this notebook publicly**, clear all cell outputs before
> committing (Colab: Edit → Clear all outputs) and double check no secret value appears
> anywhere in the saved JSON. Any key that has ever appeared in an output cell, a hardcoded
> string, or git history should be treated as compromised — rotate it in the provider's
> dashboard rather than just deleting the text.

## Known Issues

- **This run's NewsAPI key failed outright** (`API error: Your API key is invalid or
  incorrect`), returning 0 articles for all 30 days. Every day's sentiment score fell back
  to the same default value (`avg_negativity=0.2`, `sentiment_stress=0.3`), which is why
  `raw_sentiment` shows almost no feature importance — it was a constant, not a signal, for
  this run. Verify the key is current and correctly named in Colab Secrets before trusting
  the CSI's "news" component.
- **Two debug cells print secret values to output.** `userdata.get('NEWS_API_KEY')` and
  `userdata.get('WMTA_KEY')` are called as bare expressions with no assignment, so Colab
  displays the raw key value as cell output. These cells should be deleted, not just cleared,
  since re-running them regenerates the leak.
- **Secret name inconsistency:** the debug/build_transit cells read `WMTA_KEY` (missing the
  second "A"), while a leftover dead-code line reads the differently-spelled `WMATA_KEY` via
  `os.environ.get`. Confirm the Colab Secret is actually named to match whichever variable is
  live, or requests will silently succeed against the wrong (empty) key.
- **Transit data is mostly simulated.** Only "today" uses live WMATA incident/bus data; the
  other 29 days are a calibrated random simulation, not historical fact.
- **Small sample size (~30 days).** The forecasting model is deliberately kept small (6
  features, shallow trees) to avoid overfitting, but treat forecasts as directional rather
  than precise until more history is collected.

## Ideas for Extension

- Fix the NewsAPI key/secret so sentiment stops falling back to a constant every day.
- Remove the output-leaking debug cells and standardize the WMATA secret name.
- Backfill historical weather via Open-Meteo's **archive API** to get 90+ days instead of 30.
- Start logging live WMATA data daily (e.g. via GitHub Actions cron) so transit stops relying
  on simulation for 29 of 30 days.
- Swap in the NYC MTA or TfL API to extend this to another city.
- Add a Twitter/Reddit stream as a social sentiment signal.
- Deploy as a Streamlit app (e.g., on Hugging Face Spaces).

## License


