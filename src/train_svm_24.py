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

print("Loading 24-dimensional latent features...")

X_train = np.load("data/processed/X_train_latent_24.npy")
X_val = np.load("data/processed/X_val_latent_24.npy")
X_test = np.load("data/processed/X_test_latent_24.npy")

y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")
y_test = np.load("data/processed/y_test.npy")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTraining SVM on 24-D latent features...")

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

svm.fit(X_train, y_train)

print("Training completed.")

# -----------------------------
# TEST
# -----------------------------

y_test_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)

precision = precision_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

print("\n========== TEST RESULTS ==========")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Macro F1:           {macro_f1:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")

# -----------------------------
# VALIDATION
# -----------------------------

y_val_pred = svm.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)

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

print("\n========== VALIDATION RESULTS ==========")
print(f"Accuracy:           {val_accuracy:.4f}")
print(f"Macro F1:           {val_macro_f1:.4f}")
print(f"Weighted F1:        {val_weighted_f1:.4f}")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

label_encoder = joblib.load("models/label_encoder.pkl")

target_names = label_encoder.inverse_transform(
    np.arange(len(label_encoder.classes_))
)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_test_pred,
        labels=np.arange(len(target_names)),
        target_names=target_names,
        zero_division=0
    )
)

# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(
    svm,
    "models/svm_24_features.pkl"
)

print("\nModel saved to:")
print("models/svm_24_features.pkl")