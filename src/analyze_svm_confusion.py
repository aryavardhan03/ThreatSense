import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test data...")

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    f"{MODEL_DIR}/label_encoder.pkl"
)


# ============================================================
# LOAD BEST SVM
# ============================================================

model_path = (
    f"{MODEL_DIR}/svm_tuned_extended.pkl"
)

print(
    f"Loading model: {model_path}"
)

svm = joblib.load(model_path)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = svm.predict(X_test)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = label_encoder.classes_

print(
    f"\nNumber of classes: {len(class_names)}"
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
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


print("\n" + "=" * 70)
print("OVERALL TEST PERFORMANCE")
print("=" * 70)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(report).transpose()


# Remove overall summary rows for class analysis

class_report_df = report_df[
    ~report_df.index.isin(
        ["accuracy", "macro avg", "weighted avg"]
    )
].copy()


# ============================================================
# LOWEST F1 CLASSES
# ============================================================

print("\n" + "=" * 70)
print("LOWEST F1 CLASSES")
print("=" * 70)

lowest_f1 = (
    class_report_df
    .sort_values("f1-score")
    .head(10)
)

for class_name, row in lowest_f1.iterrows():

    print(
        f"{class_name:<30} "
        f"Precision={row['precision']:.3f} | "
        f"Recall={row['recall']:.3f} | "
        f"F1={row['f1-score']:.3f} | "
        f"Support={int(row['support'])}"
    )


# ============================================================
# HIGHEST F1 CLASSES
# ============================================================

print("\n" + "=" * 70)
print("HIGHEST F1 CLASSES")
print("=" * 70)

highest_f1 = (
    class_report_df
    .sort_values(
        "f1-score",
        ascending=False
    )
    .head(10)
)

for class_name, row in highest_f1.iterrows():

    print(
        f"{class_name:<30} "
        f"Precision={row['precision']:.3f} | "
        f"Recall={row['recall']:.3f} | "
        f"F1={row['f1-score']:.3f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(
        len(class_names)
    )
)


# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

confusions = []

for actual_idx in range(
    len(class_names)
):

    for predicted_idx in range(
        len(class_names)
    ):

        # Ignore correct predictions
        if actual_idx == predicted_idx:
            continue

        count = cm[
            actual_idx,
            predicted_idx
        ]

        if count > 0:

            confusions.append({
                "actual": class_names[
                    actual_idx
                ],
                "predicted": class_names[
                    predicted_idx
                ],
                "count": int(count)
            })


confusions_df = pd.DataFrame(
    confusions
)


if not confusions_df.empty:

    confusions_df = (
        confusions_df
        .sort_values(
            "count",
            ascending=False
        )
    )


# ============================================================
# TOP CONFUSIONS
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 CONFUSION PAIRS")
print("=" * 70)

for _, row in confusions_df.head(30).iterrows():

    print(
        f"{row['actual']:<30} "
        f"→ {row['predicted']:<30} "
        f"{row['count']} samples"
    )


# ============================================================
# PER-CLASS CORRECT / INCORRECT
# ============================================================

class_analysis = []

for class_idx, class_name in enumerate(
    class_names
):

    actual_count = int(
        np.sum(
            y_test == class_idx
        )
    )

    correct_count = int(
        cm[
            class_idx,
            class_idx
        ]
    )

    incorrect_count = (
        actual_count
        - correct_count
    )

    class_analysis.append({

        "class": class_name,

        "actual_samples":
            actual_count,

        "correct_predictions":
            correct_count,

        "incorrect_predictions":
            incorrect_count,

        "accuracy":
            (
                correct_count
                / actual_count
                if actual_count > 0
                else 0
            )
    })


class_analysis_df = pd.DataFrame(
    class_analysis
)


# ============================================================
# MOST DIFFICULT CLASSES BY ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("MOST DIFFICULT CLASSES")
print("=" * 70)

difficult_classes = (
    class_analysis_df
    .sort_values("accuracy")
    .head(10)
)

for _, row in difficult_classes.iterrows():

    print(
        f"{row['class']:<30} "
        f"Accuracy={row['accuracy']:.3f} | "
        f"Correct={int(row['correct_predictions'])}/"
        f"{int(row['actual_samples'])}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

report_path = (
    f"{RESULTS_DIR}/svm_classification_report.csv"
)

confusion_path = (
    f"{RESULTS_DIR}/svm_confusion_matrix.csv"
)

confusions_path = (
    f"{RESULTS_DIR}/svm_top_confusions.csv"
)

class_analysis_path = (
    f"{RESULTS_DIR}/svm_class_analysis.csv"
)


class_report_df.to_csv(
    report_path
)

pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
).to_csv(
    confusion_path
)

confusions_df.to_csv(
    confusions_path,
    index=False
)

class_analysis_df.to_csv(
    class_analysis_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles saved:")

print(report_path)
print(confusion_path)
print(confusions_path)
print(class_analysis_path)

print("=" * 70)