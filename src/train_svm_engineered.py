import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"


print("Loading engineered features...")

X_train = np.load(
    f"{DATA_DIR}/X_train_engineered.npy"
)

X_val = np.load(
    f"{DATA_DIR}/X_val_engineered.npy"
)

X_test = np.load(
    f"{DATA_DIR}/X_test_engineered.npy"
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


print("\nData shapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ============================================================
# TRAIN SVM
# ============================================================

print("\nTraining SVM...")
print("Kernel       : RBF")
print("C            : 500")
print("Gamma        : 0.02")
print("Class weight : None")


svm = SVC(
    kernel="rbf",
    C=500,
    gamma=0.02,
    class_weight=None,
    probability=True,
    random_state=42
)


svm.fit(
    X_train,
    y_train
)


print("\nSVM training completed.")


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


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

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


print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy          : {test_accuracy:.4f}")
print(f"Weighted Precision: {test_precision:.4f}")
print(f"Weighted Recall   : {test_recall:.4f}")
print(f"Macro F1          : {test_macro_f1:.4f}")
print(f"Weighted F1       : {test_weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

label_encoder = joblib.load(
    f"{MODEL_DIR}/label_encoder.pkl"
)

class_names = label_encoder.classes_


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    f"{MODEL_DIR}/svm_engineered.pkl"
)

joblib.dump(
    svm,
    model_path
)

print("\nModel saved to:")
print(model_path)

print("\n" + "=" * 60)
print("ENGINEERED SVM EXPERIMENT COMPLETED")
print("=" * 60)