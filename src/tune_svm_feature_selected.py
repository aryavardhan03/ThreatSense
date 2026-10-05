import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

# Load the feature selector from the previous experiment
selector = joblib.load(
    os.path.join(MODEL_DIR, "feature_selector.pkl")
)


# ============================================================
# APPLY FEATURE SELECTION
# ============================================================

X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)

print("=" * 70)
print("TUNING SVM ON SELECTED FEATURES")
print("=" * 70)

print(f"Original training shape   : {X_train.shape}")
print(f"Selected training shape   : {X_train_selected.shape}")
print(f"Original validation shape : {X_val.shape}")
print(f"Selected validation shape : {X_val_selected.shape}")


# ============================================================
# PARAMETER GRID
# ============================================================

C_VALUES = [
    50,
    100,
    200,
    500,
    1000,
    2000
]

GAMMA_VALUES = [
    "scale",
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
# BEST MODEL TRACKING
# ============================================================

best_macro_f1 = -1
best_accuracy = 0

best_params = None
best_model = None

results = []

total_experiments = (
    len(C_VALUES)
    * len(GAMMA_VALUES)
    * len(CLASS_WEIGHTS)
)

current_experiment = 0


# ============================================================
# GRID SEARCH
# ============================================================

for class_weight in CLASS_WEIGHTS:

    for C in C_VALUES:

        for gamma in GAMMA_VALUES:

            current_experiment += 1

            print(
                f"\n[{current_experiment}/{total_experiments}] "
                f"C={C}, gamma={gamma}, "
                f"class_weight={class_weight}"
            )

            svm = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                class_weight=class_weight,
                probability=True,
                random_state=42
            )

            svm.fit(
                X_train_selected,
                y_train
            )

            y_pred = svm.predict(
                X_val_selected
            )

            accuracy = accuracy_score(
                y_val,
                y_pred
            )

            macro_f1 = f1_score(
                y_val,
                y_pred,
                average="macro"
            )

            weighted_f1 = f1_score(
                y_val,
                y_pred,
                average="weighted"
            )

            print(
                f"Accuracy={accuracy:.4f}, "
                f"Macro F1={macro_f1:.4f}, "
                f"Weighted F1={weighted_f1:.4f}"
            )

            results.append({
                "C": C,
                "gamma": gamma,
                "class_weight": class_weight,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1
            })

            # Select based on Macro F1
            if macro_f1 > best_macro_f1:

                best_macro_f1 = macro_f1
                best_accuracy = accuracy

                best_params = {
                    "C": C,
                    "gamma": gamma,
                    "class_weight": class_weight
                }

                best_model = svm


# ============================================================
# SORT RESULTS
# ============================================================

results = sorted(
    results,
    key=lambda x: x["macro_f1"],
    reverse=True
)


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

print("\n\n")
print("=" * 90)
print("TOP 15 SVM CONFIGURATIONS")
print("=" * 90)

print(
    f"{'Rank':<6}"
    f"{'C':<8}"
    f"{'Gamma':<12}"
    f"{'Weight':<15}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Weighted F1':<12}"
)

print("-" * 90)

for rank, result in enumerate(
    results[:15],
    start=1
):

    print(
        f"{rank:<6}"
        f"{result['C']:<8}"
        f"{str(result['gamma']):<12}"
        f"{str(result['class_weight']):<15}"
        f"{result['accuracy']:<12.4f}"
        f"{result['macro_f1']:<12.4f}"
        f"{result['weighted_f1']:<12.4f}"
    )


# ============================================================
# BEST CONFIGURATION
# ============================================================

print("\n")
print("=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    f"C            : {best_params['C']}"
)

print(
    f"Gamma        : {best_params['gamma']}"
)

print(
    f"Class weight : {best_params['class_weight']}"
)

print(
    f"Validation Accuracy : {best_accuracy:.4f}"
)

print(
    f"Validation Macro F1 : {best_macro_f1:.4f}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "svm_feature_selected_tuned.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nSaved best model:")
print(model_path)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "svm_feature_selection_tuning_results.pkl"
)

joblib.dump(
    results,
    results_path
)

print("\nSaved tuning results:")
print(results_path)

print("\nSVM feature-selected tuning completed.")