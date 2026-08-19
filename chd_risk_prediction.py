"""
Predicting 10-Year CHD Risk
Logistic Regression + Framingham dataset.
Educational prototype; not a clinical diagnostic tool.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATASET = Path("framingham.csv")
COLUMNS = [
    "male","age","education","currentSmoker","cigsPerDay","BPMeds",
    "prevalentStroke","prevalentHyp","diabetes","totChol","sysBP",
    "diaBP","BMI","heartRate","glucose","TenYearCHD"
]
PLOT_FEATURES = ["age","cigsPerDay","totChol","sysBP","diaBP"]

def load_data(path=DATASET):
    if not path.exists():
        raise FileNotFoundError("Place framingham.csv in the project root.")
    data = pd.read_csv(path, na_values="NA")
    data.columns = COLUMNS
    return data.dropna()

def main():
    data = load_data()
    X = data.drop("TenYearCHD", axis=1)
    y = data["TenYearCHD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    importance = pd.Series(model.coef_[0], index=X.columns)
    print("\nTop Features:")
    print(importance.abs().sort_values(ascending=False).head(10))

    probabilities = model.predict_proba(X_test)[:, 1]
    result = X_test.copy()
    result["Predicted_Probability_CHD"] = probabilities

    out = Path("visualizations/generated")
    out.mkdir(parents=True, exist_ok=True)

    for feature in PLOT_FEATURES:
        plt.figure(figsize=(8, 6))
        plt.scatter(result[feature], result["Predicted_Probability_CHD"], alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel("Predicted Probability of 10-Year CHD")
        plt.title(f"{feature} vs. Predicted Probability of 10-Year CHD")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / f"{feature}_vs_probability.png", dpi=160)
        plt.close()

if __name__ == "__main__":
    main()
