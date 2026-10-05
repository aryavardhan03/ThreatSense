import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD
# ============================================================

MODEL_DIR = "models"
DATA_DIR = "data/processed"

model = joblib.load(
    f"{MODEL_DIR}/svm_tuned_extended.pkl"
)

label_encoder = joblib.load(
    f"{MODEL_DIR}/label_encoder.pkl"
)

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{DATA_DIR}/y_test.npy"
)

y_pred = model.predict(X_test)

class_names = label_encoder.classes_

actual_labels = label_encoder.inverse_transform(y_test)
predicted_labels = label_encoder.inverse_transform(y_pred)


# ============================================================
# DEFINE ATTACK FAMILIES
# ============================================================

def get_family(label):

    if label == "BENIGN":
        return "BENIGN"

    if label.startswith("DDOS-"):
        return "DDOS"

    if label.startswith("DOS-"):
        return "DOS"

    if label.startswith("MIRAI-"):
        return "MIRAI"

    if label.startswith("RECON-"):
        return "RECON"

    if label in [
        "SQLINJECTION",
        "COMMANDINJECTION",
        "XSS",
        "BROWSERHIJACKING",
        "UPLOADING_ATTACK"
    ]:
        return "WEB_ATTACK"

    if label in [
        "DNS_SPOOFING",
        "MITM-ARPSPOOFING"
    ]:
        return "SPOOFING_MITM"

    if label == "DICTIONARYBRUTEFORCE":
        return "BRUTE_FORCE"

    if label == "VULNERABILITYSCAN":
        return "VULNERABILITY_SCAN"

    if label == "BACKDOOR_MALWARE":
        return "MALWARE"

    return "OTHER"


# ============================================================
# BINARY CLASSIFICATION
# ============================================================

actual_binary = np.array([
    "BENIGN" if label == "BENIGN" else "ATTACK"
    for label in actual_labels
])

pred_binary = np.array([
    "BENIGN" if label == "BENIGN" else "ATTACK"
    for label in predicted_labels
])


print("\n")
print("=" * 70)
print("LEVEL 1 — BINARY CLASSIFICATION")
print("=" * 70)

print(
    classification_report(
        actual_binary,
        pred_binary,
        zero_division=0
    )
)

print(
    f"Accuracy : "
    f"{accuracy_score(actual_binary, pred_binary):.4f}"
)

print(
    f"Macro F1 : "
    f"{f1_score(actual_binary, pred_binary, average='macro'):.4f}"
)


# ============================================================
# ATTACK FAMILY
# ============================================================

actual_family = np.array([
    get_family(label)
    for label in actual_labels
])

pred_family = np.array([
    get_family(label)
    for label in predicted_labels
])


print("\n")
print("=" * 70)
print("LEVEL 2 — ATTACK FAMILY CLASSIFICATION")
print("=" * 70)

family_accuracy = accuracy_score(
    actual_family,
    pred_family
)

family_macro_f1 = f1_score(
    actual_family,
    pred_family,
    average="macro",
    zero_division=0
)

family_weighted_f1 = f1_score(
    actual_family,
    pred_family,
    average="weighted",
    zero_division=0
)

print(
    f"Accuracy          : {family_accuracy:.4f}"
)

print(
    f"Macro F1          : {family_macro_f1:.4f}"
)

print(
    f"Weighted F1       : {family_weighted_f1:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        actual_family,
        pred_family,
        zero_division=0
    )
)


# ============================================================
# FAMILY CONFUSION MATRIX
# ============================================================

families = sorted(
    set(actual_family) | set(pred_family)
)

cm = confusion_matrix(
    actual_family,
    pred_family,
    labels=families
)

print("\n")
print("=" * 70)
print("ATTACK FAMILY CONFUSION MATRIX")
print("=" * 70)

print("Families:")
print(families)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

np.save(
    f"{DATA_DIR}/actual_binary.npy",
    actual_binary
)

np.save(
    f"{DATA_DIR}/pred_binary.npy",
    pred_binary
)

np.save(
    f"{DATA_DIR}/actual_family.npy",
    actual_family
)

np.save(
    f"{DATA_DIR}/pred_family.npy",
    pred_family
)

print("\nResults saved.")


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"34-class Accuracy       : "
    f"{accuracy_score(actual_labels, predicted_labels):.4f}"
)

print(
    f"Binary Accuracy         : "
    f"{accuracy_score(actual_binary, pred_binary):.4f}"
)

print(
    f"Attack-family Accuracy  : "
    f"{family_accuracy:.4f}"
)

print(
    f"34-class Macro F1       : "
    f"{f1_score(actual_labels, predicted_labels, average='macro', zero_division=0):.4f}"
)

print(
    f"Binary Macro F1         : "
    f"{f1_score(actual_binary, pred_binary, average='macro', zero_division=0):.4f}"
)

print(
    f"Family Macro F1         : "
    f"{family_macro_f1:.4f}"
)

print("=" * 70)