# House Price Prediction

A **multivariate linear regression** project that predicts house prices using property and amenity features. Built with Python, pandas, scikit-learn, and Jupyter.

---

## Overview

This project uses a housing dataset to train a linear regression model that estimates house prices (in Indian Rupees) from features such as area, number of bedrooms/bathrooms, stories, and binary/categorical attributes (main road, guest room, basement, hot water heating, air conditioning, parking, preferred area, furnishing status).

---

## Project Structure

```
HousePricePrediction/
├── Housing.csv           # Dataset (545 samples, 13 columns)
├── multivariatereg.ipynb # Jupyter notebook: EDA, preprocessing, model, evaluation
└── README.md             # This file
```

---

## Dataset

**File:** `Housing.csv`

| Column            | Type    | Description                                      |
|-------------------|---------|--------------------------------------------------|
| `price`           | int     | Target variable — house price (₹)                |
| `area`            | int     | Area in sq ft                                    |
| `bedrooms`        | int     | Number of bedrooms                               |
| `bathrooms`       | int     | Number of bathrooms                              |
| `stories`         | int     | Number of stories/floors                         |
| `mainroad`        | object  | Connected to main road: `yes` / `no`             |
| `guestroom`       | object  | Guest room: `yes` / `no`                         |
| `basement`        | object  | Basement: `yes` / `no`                           |
| `hotwaterheating` | object  | Hot water heating: `yes` / `no`                  |
| `airconditioning` | object  | Air conditioning: `yes` / `no`                   |
| `parking`         | int     | Number of parking spaces (0–3)                   |
| `prefarea`        | object  | In preferred area: `yes` / `no`                  |
| `furnishingstatus`| object  | `furnished` / `semi-furnished` / `unfurnished`   |

- **Rows:** 545  
- **Target:** `price` (range ~₹17.5L – ₹1.33Cr in the data)

---

## Requirements

- **Python** 3.8+
- **Libraries:**  
  `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## How to Run

1. **Clone or download** this repository and ensure `Housing.csv` is in the same directory as the notebook.

2. **Rename the dataset** (if needed):  
   The notebook reads `housing.csv` (lowercase). If your file is `Housing.csv`, either:
   - Rename it to `housing.csv`, or  
   - In the notebook, change the path to `"Housing.csv"`.

3. **Start Jupyter** and open the notebook:

   ```bash
   jupyter notebook multivariatereg.ipynb
   ```

4. **Run all cells** in order (Kernel → Restart & Run All).

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Load data, inspect first rows, check for missing values.
- Use `info()` and `describe()` for dtypes and summary stats.
- Optional: visualizations with matplotlib/seaborn.

### 2. Train / Validation / Test Split

- **80%** training  
- **10%** validation (for tuning / early checks)  
- **10%** test (final evaluation)  
- `random_state=42` for reproducibility.

### 3. Feature Engineering

- **Missing values**
  - Numerical: fill with **median**
  - Categorical: fill with **mode**

- **Binary features** (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`):  
  `yes` → 1, `no` → 0.

- **Categorical** (`furnishingstatus`):  
  **One-hot encoding** with `drop="first"` to avoid multicollinearity (e.g. `semi-furnished`, `unfurnished`).

- **Numerical features** (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`):  
  **StandardScaler** (zero mean, unit variance) on train; same transform applied to validation and test.

### 4. Model

- **Algorithm:** `sklearn.linear_model.LinearRegression`
- **Input:** Preprocessed feature matrix (numerical + binary + one-hot).
- **Output:** Predicted house price (₹).

### 5. Evaluation

Metrics used:

- **MAE** — Mean Absolute Error (₹)
- **RMSE** — Root Mean Squared Error (₹)
- **R²** — Coefficient of determination

Reported on **validation** and **test** sets.

---

## Results (from notebook)

| Set        | MAE (₹)   | RMSE (₹)   | R²    |
|-----------|-----------|------------|--------|
| Validation| ~917,948  | ~1,305,190 | ~0.63  |
| Test      | ~1,021,192| ~1,343,202 | ~0.67  |

The model explains about **63–67%** of the variance in price; remaining error can be reduced with more features, non-linear models, or feature engineering.

---

## Making a Prediction

The notebook includes an example of predicting the price for a **new house**:

- Define a row with: `area`, `bedrooms`, `bathrooms`, `stories`, binary flags (0/1), `parking`, and `furnishingstatus`.
- Apply the **same preprocessing**: binary mapping, one-hot for `furnishingstatus`, and scaling numerical features with the **fitted** `StandardScaler` and `OneHotEncoder`.
- Call `model.predict(new_house)` to get the predicted price (₹).

Example (conceptually): a 3000 sq ft, 3BHK, 2 bathrooms, 2 stories, with main road, basement, AC, 2 parking, preferred area, furnished → predicted price in the notebook is around **₹68.3 Lakh**.

---

## Possible Next Steps

- Try **Ridge** or **Lasso** regression for regularization.
- Add **polynomial or interaction features** and compare R²/RMSE.
- Use **other models** (e.g. Random Forest, XGBoost) and compare.
- **Feature selection** (e.g. RFE, correlation analysis) to simplify the model.
- **Cross-validation** for more robust performance estimates.

---

## License

This project is for educational use. The housing dataset may have its own terms of use depending on source.
