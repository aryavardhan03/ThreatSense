import time
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"


# ============================================================
# LOAD ORIGINAL 39 FEATURES
# ============================================================

print("Loading original 39 features...")

X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
)

X_val = np.load(
    f"{DATA_DIR}/X_val.npy"
)

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
)

y_val = np.load(
    f"{DATA_DIR}/y_val.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)


print("\nShapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ============================================================
# HYPERPARAMETER GRID
# ============================================================
#
# We are expanding around the previous best:
#
# Previous best:
# C = 100
# gamma = scale
#
# Now we investigate whether larger C values and explicit
# gamma values can improve the decision boundaries.
# ============================================================

C_VALUES = [
    10,
    25,
    50,
    100,
    200,
    500,
    1000
]

GAMMA_VALUES = [
    "scale",
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1
]

CLASS_WEIGHTS = [
    None,
    "balanced"
]


# ============================================================
# STORAGE
# ============================================================

results = []

best_macro_f1 = -1
best_config = None


# ============================================================
# GRID SEARCH
# ============================================================

total_experiments = (
    len(C_VALUES)
    * len(GAMMA_VALUES)
    * len(CLASS_WEIGHTS)
)

experiment_number = 0

print("\n" + "=" * 70)
print("EXTENDED SVM HYPERPARAMETER SEARCH")
print("=" * 70)

print(
    f"Total configurations: {total_experiments}"
)

print("=" * 70)


for class_weight in CLASS_WEIGHTS:

    for C in C_VALUES:

        for gamma in GAMMA_VALUES:

            experiment_number += 1

            print(
                f"\n[{experiment_number}/"
                f"{total_experiments}] "
                f"C={C}, gamma={gamma}, "
                f"class_weight={class_weight}"
            )

            start_time = time.time()

            svm = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                class_weight=class_weight,
                probability=False,
                random_state=42
            )

            svm.fit(
                X_train,
                y_train
            )

            y_val_pred = svm.predict(
                X_val
            )

            accuracy = accuracy_score(
                y_val,
                y_val_pred
            )

            precision = precision_score(
                y_val,
                y_val_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_val,
                y_val_pred,
                average="weighted",
                zero_division=0
            )

            macro_f1 = f1_score(
                y_val,
                y_val_pred,
                average="macro",
                zero_division=0
            )

            weighted_f1 = f1_score(
                y_val,
                y_val_pred,
                average="weighted",
                zero_division=0
            )

            elapsed = time.time() - start_time

            results.append({
                "C": C,
                "gamma": gamma,
                "class_weight": class_weight,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1
            })

            print(
                f"Accuracy={accuracy:.4f} | "
                f"Macro F1={macro_f1:.4f} | "
                f"Weighted F1={weighted_f1:.4f} | "
                f"Time={elapsed:.1f}s"
            )

            # ------------------------------------------------
            # Track best configuration using VALIDATION
            # Macro F1 is the primary metric because we have
            # 34 attack classes.
            # ------------------------------------------------

            if macro_f1 > best_macro_f1:

                best_macro_f1 = macro_f1

                best_config = {
                    "C": C,
                    "gamma": gamma,
                    "class_weight": class_weight
                }

                print(
                    ">>> NEW BEST CONFIGURATION"
                )
                print(
                    f">>> Validation Macro F1: "
                    f"{best_macro_f1:.4f}"
                )


# ============================================================
# SORT RESULTS
# ============================================================

results_sorted = sorted(
    results,
    key=lambda x: x["macro_f1"],
    reverse=True
)


# ============================================================
# PRINT TOP RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 VALIDATION CONFIGURATIONS")
print("=" * 70)

for i, result in enumerate(
    results_sorted[:10],
    start=1
):

    print(
        f"{i:2d}. "
        f"C={result['C']:<4} | "
        f"gamma={str(result['gamma']):<7} | "
        f"class_weight={str(result['class_weight']):<8} | "
        f"Accuracy={result['accuracy']:.4f} | "
        f"Macro F1={result['macro_f1']:.4f} | "
        f"Weighted F1={result['weighted_f1']:.4f}"
    )


# ============================================================
# BEST CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("BEST VALIDATION CONFIGURATION")
print("=" * 70)

print(
    f"C            : {best_config['C']}"
)

print(
    f"Gamma        : {best_config['gamma']}"
)

print(
    f"Class weight : {best_config['class_weight']}"
)

print(
    f"Validation Macro F1: "
    f"{best_macro_f1:.4f}"
)


# ============================================================
# FINAL MODEL
# ============================================================
#
# IMPORTANT:
# The test set is used ONLY here, after selecting the
# configuration using validation performance.
# ============================================================

print("\nTraining final SVM using best configuration...")

final_svm = SVC(
    kernel="rbf",
    C=best_config["C"],
    gamma=best_config["gamma"],
    class_weight=best_config["class_weight"],
    probability=True,
    random_state=42
)

final_svm.fit(
    X_train,
    y_train
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating final model on TEST set...")

y_test_pred = final_svm.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy           : {test_accuracy:.4f}"
)

print(
    f"Weighted Precision : {test_precision:.4f}"
)

print(
    f"Weighted Recall    : {test_recall:.4f}"
)

print(
    f"Macro F1           : {test_macro_f1:.4f}"
)

print(
    f"Weighted F1        : {test_weighted_f1:.4f}"
)

print("=" * 70)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model_path = (
    f"{MODEL_DIR}/svm_tuned_extended.pkl"
)

joblib.dump(
    final_svm,
    model_path
)

print(
    f"\nFinal model saved to:"
    f"\n{model_path}"
)