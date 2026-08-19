# Predicting 10-Year CHD Risk

A machine-learning project using **Logistic Regression** and the **Framingham Heart Study dataset** to estimate 10-year Coronary Heart Disease (CHD) risk.

## What the project does

1. Loads `framingham.csv`.
2. Removes rows containing missing values.
3. Separates predictors from the `TenYearCHD` target.
4. Uses a 75/25 train-test split (`random_state=42`).
5. Trains a Logistic Regression classifier.
6. Produces predictions and predicted probabilities.
7. Evaluates accuracy, confusion matrix and classification report.
8. Examines coefficient-based feature importance.
9. Generates probability visualizations for age, cigarettes/day, cholesterol, systolic BP and diastolic BP.

## Reported result

The supplied project report records **84.8% overall accuracy**. It also reports **99% recall for the No-CHD class** but only **8% recall for the CHD class**. Therefore, the current model should **not** be presented as a reliable real-world screening or diagnostic system.

## Project structure

```text
chd-risk-prediction/
├── src/
│   └── chd_risk_prediction.py
├── visualizations/
├── docs/
│   ├── CHD_Risk_Prediction_Project.pdf
│   └── original_untitled0.py
├── framingham.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone <YOUR-REPOSITORY-URL>
cd chd-risk-prediction
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Place `framingham.csv` in the repository root, then run:

```bash
python src/chd_risk_prediction.py
```

Generated plots are saved under `visualizations/generated/`.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Logistic Regression

## Limitations

The supplied evaluation shows severe class-imbalance effects and low recall for positive CHD cases. Future work should investigate class balancing, threshold tuning, feature scaling/selection, calibration, cross-validation and alternative models before considering any clinical application.

## Author

**Dhrubajyoti Das**  
AI / Machine Learning Student & Developer
