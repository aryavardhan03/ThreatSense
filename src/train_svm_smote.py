import os
import numpy as np
import joblib

from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC
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
print("SMOTE + SVM EXPERIMENT")
print("=" * 75)

print(
    f"Original training shape : {X_train.shape}"
)

print(
    f"Selected training shape : {X_train_selected.shape}"
)

print(
    f"Validation shape        : {X_val_selected.shape}"
)


# ============================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

unique_before, counts_before = np.unique(
    y_train,
    return_counts=True
)

print("\nOriginal training class distribution:")

print(
    f"Minimum class size: {counts_before.min()}"
)

print(
    f"Maximum class size: {counts_before.max()}"
)


# ============================================================
# SMOTE
# ============================================================

print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_selected,
    y_train
)


# ============================================================
# SMOTE DISTRIBUTION
# ============================================================

unique_after, counts_after = np.unique(
    y_train_smote,
    return_counts=True
)

print("\nAfter SMOTE:")

print(
    f"Training shape: {X_train_smote.shape}"
)

print(
    f"Minimum class size: {counts_after.min()}"
)

print(
    f"Maximum class size: {counts_after.max()}"
)


# ============================================================
# SVM EXPERIMENTS
# ============================================================

configs = [
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
    }
]


results = []

best_macro_f1 = -1
best_accuracy = 0
best_config = None
best_model = None


# ============================================================
# TRAIN SVMs
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

    svm = SVC(
        kernel="rbf",
        C=config["C"],
        gamma=config["gamma"],
        class_weight=None,
        probability=True,
        random_state=42
    )

    svm.fit(
        X_train_smote,
        y_train_smote
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

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_accuracy = accuracy
        best_config = config
        best_model = svm


# ============================================================
# RESULTS
# ============================================================

print("\n\n")
print("=" * 75)
print("SMOTE + SVM RESULTS")
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
print("BEST SMOTE + SVM MODEL")
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
    "svm_smote.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nSaved:")
print(model_path)

print("\nSMOTE + SVM experiment completed.")