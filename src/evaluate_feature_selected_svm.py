import os
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

# ============================================================
# LOAD FEATURE SELECTOR + BEST SVM
# ============================================================

selector = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_selector.pkl"
    )
)

svm = joblib.load(
    os.path.join(
        MODEL_DIR,
        "svm_feature_selected_tuned.pkl"
    )
)

# ============================================================
# APPLY FEATURE SELECTION
# ============================================================

X_test_selected = selector.transform(X_test)

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print(f"Original test shape  : {X_test.shape}")
print(f"Selected test shape  : {X_test_selected.shape}")

# ============================================================
# PREDICTION
# ============================================================

y_pred = svm.predict(
    X_test_selected
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall   : {recall:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

label_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)

class_names = label_encoder.classes_

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

report_path = os.path.join(
    RESULTS_DIR,
    "svm_feature_selected_test_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("FINAL TEST EVALUATION\n")
    f.write("=" * 70 + "\n")

    f.write(
        f"Accuracy           : {accuracy:.4f}\n"
    )

    f.write(
        f"Weighted Precision : {precision:.4f}\n"
    )

    f.write(
        f"Weighted Recall    : {recall:.4f}\n"
    )

    f.write(
        f"Macro F1           : {macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1        : {weighted_f1:.4f}\n"
    )

    f.write("\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(report)

print("\nSaved:")
print(report_path)

print("\nFinal test evaluation completed.")