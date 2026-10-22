"""
AI-Based Overvoltage and Undervoltage Condition Classification
Educational simulation only. Thresholds are illustrative for a simulated 12 V DC system.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42

def classify_voltage(v):
    if v < 10.8:
        return "Undervoltage"
    if v > 13.2:
        return "Overvoltage"
    return "Normal"

def main():
    np.random.seed(RANDOM_STATE)
    n = 3000
    voltage = np.random.uniform(9.0, 15.0, n)
    current = np.random.uniform(0.1, 2.0, n)
    load = np.random.uniform(10, 100, n)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="min")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "voltage": voltage.round(3),
        "current": current.round(3),
        "load": load.round(2)
    })
    df["condition"] = df["voltage"].apply(classify_voltage)
    df.to_csv("voltage_dataset.csv", index=False)

    print("Dataset preview:")
    print(df.head())
    print("\nClass counts:\n", df["condition"].value_counts())
    print("\nMissing values:\n", df.isnull().sum())

    df["voltage"].hist(bins=30)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Frequency")
    plt.title("Simulated Voltage Distribution")
    plt.tight_layout()
    plt.savefig("voltage_distribution.png", dpi=160)
    plt.close()

    X = df[["voltage", "current", "load"]]
    y = df["condition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150, random_state=RANDOM_STATE, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

    labels = ["Undervoltage", "Normal", "Overvoltage"]
    cm = confusion_matrix(y_test, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump(model, "voltage_model.pkl")
    with open("evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_test, pred, zero_division=0))
        f.write("\nConfusion matrix (Undervoltage, Normal, Overvoltage):\n")
        f.write(str(cm))
    print("\nSaved voltage_dataset.csv, voltage_model.pkl, graphs, and evaluation_summary.txt")

if __name__ == "__main__":
    main()
