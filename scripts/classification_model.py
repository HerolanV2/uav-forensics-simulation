
"""
classification_model.py
Author: Hakan Emre Erolan
Description: Classification of drone flight data into normal, GPS spoofing, and RF jamming using ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

# --- 1. Load Dataset ---
df = pd.read_csv(DATASETS_DIR / "drone_simulation_dataset.csv", parse_dates=["timestamp"])
print("Dataset loaded:", df.shape)
print(df.head())

# --- 2. Features and Labels Split ---
X = df.drop(columns=["label", "timestamp", "spoofing_flag", "jamming_flag"])
y = df["label"]

# --- 3. Label Encoding ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
df["label_encoded"] = y_encoded

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 5. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Models ---
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = []
accuracy_scores = {}

# --- 7. Training & Evaluation ---
for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    print(f"\n{name} Report:\n", report)

    # Add report metrics to results table
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    precision = report_dict["weighted avg"]["precision"]
    recall = report_dict["weighted avg"]["recall"]
    f1 = report_dict["weighted avg"]["f1-score"]
    results.append([name, acc, precision, recall, f1])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name.replace(' ','_')}_confusion_matrix.png")
    plt.close()

# --- 8. Results Table ---
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n Model Comparison:\n")
print(results_df)

results_df.to_csv(RESULTS_DIR / "classification_results.csv", index=False)
print("\n Results saved as classification_results.csv")

# --- 9. Export row-level ML predictions for timeline ---

best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
best_model = models[best_model_name]

print(f"\n Best model selected for predictions: {best_model_name}")

X_full = df.drop(columns=["label", "label_encoded", "timestamp", "spoofing_flag", "jamming_flag"])
X_full_scaled = scaler.transform(X_full)
full_preds = best_model.predict(X_full_scaled)
full_labels = encoder.inverse_transform(full_preds)

ml_pred_df = pd.DataFrame({
    "timestamp": df["timestamp"],
    "predicted_label": full_labels
})

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ml_pred_df.to_csv(RESULTS_DIR / "ml_predictions.csv", index=False)
print("ML predictions saved to results/ml_predictions.csv")

