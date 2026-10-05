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


# --------------------------------------------------
# 1. Load latent features
# --------------------------------------------------

X_train = np.load("data/processed/X_train_latent.npy")
X_val = np.load("data/processed/X_val_latent.npy")
X_test = np.load("data/processed/X_test_latent.npy")


# --------------------------------------------------
# 2. Load encoded labels
# --------------------------------------------------

y_train = np.load("data/processed/y_train.npy")
y_val = np.load("data/processed/y_val.npy")
y_test = np.load("data/processed/y_test.npy")


print("Latent feature shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nLabel shapes:")
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 3. Create SVM classifier
# --------------------------------------------------

print("\nCreating SVM classifier...")

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True
)


# --------------------------------------------------
# 4. Train SVM
# --------------------------------------------------

print("\nStarting SVM training...")

svm_model.fit(
    X_train,
    y_train
)

print("\nSVM training completed successfully.")


# --------------------------------------------------
# 5. Validation prediction
# --------------------------------------------------

print("\nEvaluating on validation data...")

y_val_pred = svm_model.predict(X_val)


val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

val_macro_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro"
)

val_weighted_f1 = f1_score(
    y_val,
    y_val_pred,
    average="weighted"
)


print("\nValidation Results:")
print("Accuracy:", round(val_accuracy, 4))
print("Macro F1:", round(val_macro_f1, 4))
print("Weighted F1:", round(val_weighted_f1, 4))


# --------------------------------------------------
# 6. Test prediction
# --------------------------------------------------

print("\nEvaluating on test data...")

y_test_pred = svm_model.predict(X_test)


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


print("\nTest Results:")
print("Accuracy:", round(test_accuracy, 4))
print("Weighted Precision:", round(test_precision, 4))
print("Weighted Recall:", round(test_recall, 4))
print("Macro F1:", round(test_macro_f1, 4))
print("Weighted F1:", round(test_weighted_f1, 4))


# --------------------------------------------------
# 7. Detailed classification report
# --------------------------------------------------

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# --------------------------------------------------
# 8. Save SVM model
# --------------------------------------------------

joblib.dump(
    svm_model,
    "models/svm_classifier.pkl"
)

print("\nSVM model saved successfully:")
print("models/svm_classifier.pkl")