# 🚲 Bike Sharing Demand Prediction

A reproducible **calendar-only** machine-learning regression portfolio project using the UCI Bike Sharing hourly dataset (2011–2012, 17,379 records).

## Results (held-out chronological test)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random forest | **63.434** | **98.176** | **0.7917** |

The model was selected by validation MAE, refitted on the first 85% of chronological data, then evaluated once on the final 15%. The test metrics are **historical**, not evidence of current-day forecast accuracy.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.model
streamlit run app.py
pytest -q
```

## Repository layout

- `data/hour.csv`: UCI Bike Sharing hourly data
- `src/model.py`: fixed, reproducible random-forest training/evaluation
- `app.py`: Streamlit historical-scenario interface
- `artifacts/`: real model metrics, plots, SHAP stability experiment
- `reports/technical_report.md`: methodology, diagnostics, limitations
- `tests/`: data and model sanity checks

## Responsible interpretation

This is a **historical research demo** restricted to 2011–2012. No observed weather, `casual`, or `registered` fields are used as inputs. It is **not** a live forecast, nor suitable for station-level operational decisions. Holiday status must be supplied using the dataset's calendar definition. SHAP uses an approximate calculation over 4 samples of 24 records; this does not establish future-year stability.

Dataset: Fanaee-T, H. & Gama, J. (2014), *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence. UCI Machine Learning Repository, Bike Sharing dataset. See https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset . Verify dataset licensing and attribution requirements before publishing.
