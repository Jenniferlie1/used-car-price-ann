# Used Car Price Prediction with ANN

Predicting used car selling prices (`selling_price`) with an Artificial Neural Network (Keras): EDA, preprocessing, a baseline model, a modified model, and evaluation. Built as a Deep Learning course assignment.

## Dataset

`1B.parquet` — 2,000 rows, 18 columns (brand, year, km driven, fuel, transmission, owner, engine, max power, torque, seats, etc.). Target: `selling_price`. The dataset file is not included in this repository.

## Preprocessing

- Standardized `mileage` (comma → dot, then cast to float) and extracted the main numeric value from `torque`
- Removed the invalid `year = 3011` record and rows with missing values (under 1% of the data)
- Applied a `log1p` transform to the target because of its right-skewed distribution
- Feature selection based on Pearson correlation and mean price per category; dropped `Sales_ID`, `City`, `State or Province`, `mileage`, `Region`, and `sold`
- 70:10:20 train/validation/test split (1,393 / 199 / 399 rows), `random_state=42`
- `RobustScaler` for numerical features and `OneHotEncoder` for categorical features, fitted on the training set only (46 features in total)

## Models

| | Architecture | Training |
|---|---|---|
| Baseline | 46 → 92 → 92 → 1 (ReLU), 12,973 parameters | Adam, batch size 32, 20 epochs |
| Improved | 46 → 128 → 64 → 1 (ReLU), 14,337 parameters | Adam (lr 0.001), batch size 16, up to 100 epochs, EarlyStopping (patience 3, restore best weights) |

## Results (test set, USD scale)

| Model | MAE | RMSE | MSE | R² |
|-------|-----|------|-----|----|
| Baseline | 1227.10 | 3626.49 | 1.31 × 10⁷ | 0.8181 |
| Improved | 977.13 | 2272.33 | 5.16 × 10⁶ | 0.9286 |

The improved model performs better on every metric. Predictions are converted back to USD with `expm1` before the metrics are computed.

## Notes

- The `cap_outliers` function (IQR capping) is defined in the notebook but never called, so outliers are detected but not capped.
- `max_power` still contains an invalid negative value (min −100) that is not handled.
- Evaluation uses a single split and a single run, so the baseline vs. improved gap has not been tested against run-to-run variance.
