import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_val.npy")
)

# ============================================================
# LOAD FEATURE SELECTOR
# ============================================================

selector = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_selector.pkl"
    )
)

X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)


print("=" * 75)
print("ONE-VS-REST SVM EXPERIMENT")
print("=" * 75)

print(
    f"Training shape   : {X_train_selected.shape}"
)

print(
    f"Validation shape : {X_val_selected.shape}"
)


# ============================================================
# CONFIGURATIONS
# ============================================================

configs = [
    {
        "C": 50,
        "gamma": "scale"
    },
    {
        "C": 100,
        "gamma": "scale"
    },
    {
        "C": 200,
        "gamma": "scale"
    },
    {
        "C": 500,
        "gamma": "scale"
    },
    {
        "C": 500,
        "gamma": 0.01
    },
    {
        "C": 500,
        "gamma": 0.02
    },
    {
        "C": 500,
        "gamma": 0.05
    },
    {
        "C": 1000,
        "gamma": 0.01
    },
    {
        "C": 1000,
        "gamma": 0.02
    },
    {
        "C": 2000,
        "gamma": 0.01
    }
]


results = []

best_macro_f1 = -1
best_accuracy = 0
best_config = None
best_model = None


# ============================================================
# TRAIN ONE-VS-REST SVM
# ============================================================

for i, config in enumerate(configs, start=1):

    print("\n" + "=" * 75)

    print(
        f"Experiment {i}/{len(configs)}"
    )

    print(
        f"C={config['C']}, "
        f"gamma={config['gamma']}"
    )

    print("=" * 75)

    # Base binary SVM
    base_svm = SVC(
        kernel="rbf",
        C=config["C"],
        gamma=config["gamma"],
        class_weight=None,
        probability=True,
        random_state=42
    )

    # One-vs-Rest wrapper
    ovr_svm = OneVsRestClassifier(
        base_svm,
        n_jobs=-1
    )

    print("Training One-vs-Rest SVM...")

    ovr_svm.fit(
        X_train_selected,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = ovr_svm.predict(
        X_val_selected
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    results.append({
        "C": config["C"],
        "gamma": config["gamma"],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

    # --------------------------------------------------------
    # Track best model
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_accuracy = accuracy
        best_config = config
        best_model = ovr_svm


# ============================================================
# RESULTS
# ============================================================

print("\n\n")
print("=" * 75)
print("ONE-VS-REST SVM RESULTS")
print("=" * 75)

print(
    f"{'C':<10}"
    f"{'Gamma':<12}"
    f"{'Accuracy':<15}"
    f"{'Macro F1':<15}"
    f"{'Weighted F1':<15}"
)

print("-" * 75)

for result in results:

    print(
        f"{result['C']:<10}"
        f"{str(result['gamma']):<12}"
        f"{result['accuracy']:<15.4f}"
        f"{result['macro_f1']:<15.4f}"
        f"{result['weighted_f1']:<15.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

print("\n")
print("=" * 75)
print("BEST ONE-VS-REST MODEL")
print("=" * 75)

print(
    f"C            : {best_config['C']}"
)

print(
    f"Gamma        : {best_config['gamma']}"
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
    "svm_ovr.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nSaved:")
print(model_path)

# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "svm_ovr_results.pkl"
)

joblib.dump(
    results,
    results_path
)

print(
    results_path
)

print("\nOne-vs-Rest SVM experiment completed.")