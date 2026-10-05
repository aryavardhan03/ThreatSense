import os
import time
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD β-VAE FEATURES
# ============================================================

print("Loading β-VAE latent features...")

X_train = np.load(
    f"{DATA_DIR}/X_train_vae_beta_24.npy"
)

X_val = np.load(
    f"{DATA_DIR}/X_val_vae_beta_24.npy"
)

X_test = np.load(
    f"{DATA_DIR}/X_test_vae_beta_24.npy"
)


# ============================================================
# LOAD LABELS
# ============================================================

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
)

y_val = np.load(
    f"{DATA_DIR}/y_val.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)


print("\nData shapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print("\ny_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)


# ============================================================
# CREATE SVM
# ============================================================
#
# Same tuned configuration used for the original
# 39-feature SVM baseline.
#
# Original tuned SVM:
# C = 100
# gamma = scale
# kernel = RBF
# class_weight = balanced
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SVM ON β-VAE 24-D FEATURES")
print("=" * 60)

svm = SVC(
    kernel="rbf",
    C=100,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

start_time = time.time()

svm.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# VALIDATION
# ============================================================

print("\nEvaluating on validation set...")

y_val_pred = svm.predict(X_val)


val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

val_precision = precision_score(
    y_val,
    y_val_pred,
    average="weighted",
    zero_division=0
)

val_recall = recall_score(
    y_val,
    y_val_pred,
    average="weighted",
    zero_division=0
)

val_macro_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro",
    zero_division=0
)

val_weighted_f1 = f1_score(
    y_val,
    y_val_pred,
    average="weighted",
    zero_division=0
)


print("\nValidation Results")
print("-" * 40)
print(f"Accuracy          : {val_accuracy:.4f}")
print(f"Weighted Precision: {val_precision:.4f}")
print(f"Weighted Recall   : {val_recall:.4f}")
print(f"Macro F1          : {val_macro_f1:.4f}")
print(f"Weighted F1       : {val_weighted_f1:.4f}")


# ============================================================
# TEST
# ============================================================

print("\nEvaluating on test set...")

y_test_pred = svm.predict(X_test)


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
# PRINT TEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("β-VAE 24-D SVM — TEST RESULTS")
print("=" * 60)

print(f"Accuracy          : {test_accuracy:.4f}")
print(f"Weighted Precision: {test_precision:.4f}")
print(f"Weighted Recall   : {test_recall:.4f}")
print(f"Macro F1          : {test_macro_f1:.4f}")
print(f"Weighted F1       : {test_weighted_f1:.4f}")

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    f"{MODEL_DIR}/svm_vae_beta_24.pkl"
)

joblib.dump(
    svm,
    model_path
)

print("\nModel saved:")
print(model_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Features : β-VAE latent representation")
print("Dimensions:", X_train.shape[1])
print("Kernel   : RBF")
print("C        : 100")
print("Gamma    : scale")

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Macro F1 : {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")

print("=" * 60)