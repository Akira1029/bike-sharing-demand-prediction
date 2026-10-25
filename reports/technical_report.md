# Bike Sharing Demand Prediction — Technical Report

## Executive summary

This project estimates system-wide hourly bicycle rental demand from calendar features using 17,379 observations from the UCI Bike Sharing dataset (2011–2012). A random forest selected on a chronological validation split achieved held-out test MAE **63.434**, RMSE **98.176**, and R² **0.7917**. This is a historical benchmark, not an operational forecast.

## Problem and dataset

Predict hourly `cnt` from information available before the target hour: `yr`, `mnth`, `season`, `hr`, `weekday`, `holiday`, `workingday`. Exclude `casual` and `registered` because their sum equals the target, and exclude realized weather to avoid a mismatch with pre-hour forecast availability. `instant` is an identifier, not a predictive input.

EDA found no nulls or duplicate rows; 165 hourly timestamps are absent. Missing timestamps are not imputed as zero-demand observations.

## Evaluation design

Sort by actual hourly timestamp. Chronological 70% train / 15% validation / 15% test. Compare mean baseline, linear regression, random forest, and histogram gradient boosting using validation MAE. Random forest was chosen with 120 trees, `min_samples_leaf=3`, `max_features=0.8`, `random_state=42`. Refit on first 85%, evaluate once on final 15%.

Validation MAEs: mean 194.135, linear regression 115.732, random forest 59.648, gradient boosting 64.369. Final random forest test MAE 63.434, RMSE 98.176, R² 0.7917. Model comparison and final evaluation occur in different time periods; the selected model is also refitted on more data before final testing.

## Diagnostics

Evening peak (16–19h) test MAE **134.837** versus **44.999** at other hours and **71.610** at morning peak (07–09h). At observed demand ≥430 rentals/hour, MAE **102.77** and signed bias **−42.30**, indicating underestimation in high-demand cases. These are **descriptive test-set diagnostics**, not a basis for hyperparameter tuning.

## Explainability

Permutation importance on validation: shuffling `hr` increases MAE by **172.587**; `workingday` by **34.316**. Prior exact SHAP on 32 observations suggested `hr` was most influential. To assess sampling sensitivity, a repeated **approximate** Tree SHAP experiment sampled 4 groups of 24 observations from train+validation only. Mean absolute group contributions (mean ± sample standard deviation): `hr` 118.738 ± 18.040; `yr` 47.448 ± 3.903; `season` 41.364 ± 6.282; `workingday` 34.752 ± 11.867. The hour feature dominates these four samples, but this small approximate experiment does not prove stability on new years or causal influence. Different SHAP computation modes should not be compared numerically without caution.

## Limitations and deployment

Only 2011–2012 historical records; `yr` encodes just two observed years. Model cannot be safely extrapolated to 2026. It has no forecast weather inputs, station-level demand, or fleet availability. Validation is time-ordered but does not provide a multi-year rolling backtest. A real deployment would require current data, prospective testing, weather forecasts available at prediction time, operational monitoring, and decision-specific cost metrics.

## Reproducibility

Install `requirements.txt`, run `python -m src.model`, `pytest -q`, and `streamlit run app.py`. See artifacts for model metrics and SHAP stability outputs. All reported numbers are from prior executed experiments; no new benchmark is claimed for the demo interface.
