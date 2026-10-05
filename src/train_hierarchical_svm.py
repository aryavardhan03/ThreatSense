import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


DATA_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(f"{DATA_DIR}/X_train.npy")
X_val = np.load(f"{DATA_DIR}/X_val.npy")
X_test = np.load(f"{DATA_DIR}/X_test.npy")

y_train = np.load(f"{DATA_DIR}/y_train.npy")
y_val = np.load(f"{DATA_DIR}/y_val.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")

label_encoder = joblib.load(
    f"{MODEL_DIR}/label_encoder.pkl"
)

class_names = label_encoder.classes_


train_labels = label_encoder.inverse_transform(y_train)
val_labels = label_encoder.inverse_transform(y_val)
test_labels = label_encoder.inverse_transform(y_test)


# ============================================================
# ATTACK FAMILY MAPPING
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
# CREATE FAMILY LABELS
# ============================================================

train_families = np.array([
    get_family(x) for x in train_labels
])

val_families = np.array([
    get_family(x) for x in val_labels
])

test_families = np.array([
    get_family(x) for x in test_labels
])


# ============================================================
# TRAIN FAMILY CLASSIFIER
# ============================================================

print("=" * 70)
print("HIERARCHICAL SVM")
print("=" * 70)

print("\nTraining family-level classifier...")


family_classes = sorted(
    np.unique(train_families)
)

family_encoder = {
    family: i
    for i, family in enumerate(family_classes)
}

y_train_family = np.array([
    family_encoder[x]
    for x in train_families
])

y_val_family = np.array([
    family_encoder[x]
    for x in val_families
])

y_test_family = np.array([
    family_encoder[x]
    for x in test_families
])


family_svm = SVC(
    kernel="rbf",
    C=500,
    gamma=0.02,
    class_weight=None,
    probability=True,
    random_state=42
)

family_svm.fit(
    X_train,
    y_train_family
)


# ============================================================
# FAMILY PREDICTION
# ============================================================

val_family_pred = family_svm.predict(X_val)
test_family_pred = family_svm.predict(X_test)


print("\nFamily-level results:")

print(
    f"Validation Accuracy : "
    f"{accuracy_score(y_val_family, val_family_pred):.4f}"
)

print(
    f"Validation Macro F1 : "
    f"{f1_score(y_val_family, val_family_pred, average='macro'):.4f}"
)

print(
    f"Test Accuracy       : "
    f"{accuracy_score(y_test_family, test_family_pred):.4f}"
)

print(
    f"Test Macro F1       : "
    f"{f1_score(y_test_family, test_family_pred, average='macro'):.4f}"
)


# ============================================================
# TRAIN ONE SVM PER FAMILY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FAMILY-SPECIFIC CLASSIFIERS")
print("=" * 70)


family_models = {}

for family in family_classes:

    if family == "BENIGN":
        continue

    print(f"\nTraining classifier for: {family}")

    mask = train_families == family

    X_family = X_train[mask]
    y_family_labels = train_labels[mask]

    unique_classes = np.unique(
        y_family_labels
    )

    print(
        f"Samples : {len(X_family)}"
    )

    print(
        f"Classes : {len(unique_classes)}"
    )

    # If family contains only one attack class,
    # no SVM is required.
    if len(unique_classes) == 1:

        family_models[family] = {
            "type": "single_class",
            "class": unique_classes[0]
        }

        print(
            f"Single class → {unique_classes[0]}"
        )

        continue

    local_encoder = {
        label: i
        for i, label in enumerate(unique_classes)
    }

    y_family_encoded = np.array([
        local_encoder[x]
        for x in y_family_labels
    ])

    model = SVC(
        kernel="rbf",
        C=500,
        gamma="scale",
        class_weight=None,
        probability=True,
        random_state=42
    )

    model.fit(
        X_family,
        y_family_encoded
    )

    family_models[family] = {
        "type": "svm",
        "model": model,
        "encoder": local_encoder,
        "classes": unique_classes
    }

    print("Completed.")


# ============================================================
# HIERARCHICAL PREDICTION FUNCTION
# ============================================================

def hierarchical_predict(X):

    family_pred_encoded = family_svm.predict(X)

    final_predictions = []

    for i, family_id in enumerate(
        family_pred_encoded
    ):

        family = family_classes[family_id]

        # ----------------------------------------------------
        # BENIGN
        # ----------------------------------------------------

        if family == "BENIGN":

            final_predictions.append(
                "BENIGN"
            )

            continue

        # ----------------------------------------------------
        # FAMILY WITH SINGLE CLASS
        # ----------------------------------------------------

        info = family_models[family]

        if info["type"] == "single_class":

            final_predictions.append(
                info["class"]
            )

            continue

        # ----------------------------------------------------
        # FAMILY-SPECIFIC CLASSIFIER
        # ----------------------------------------------------

        local_model = info["model"]
        local_encoder = info["encoder"]
        local_classes = info["classes"]

        local_prediction = local_model.predict(
            X[i].reshape(1, -1)
        )[0]

        reverse_encoder = {
            value: key
            for key, value in local_encoder.items()
        }

        predicted_class = reverse_encoder[
            local_prediction
        ]

        final_predictions.append(
            predicted_class
        )

    return np.array(final_predictions)


# ============================================================
# HIERARCHICAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL VALIDATION")
print("=" * 70)

val_hierarchical_pred = hierarchical_predict(
    X_val
)

val_hierarchical_encoded = label_encoder.transform(
    val_hierarchical_pred
)

print(
    f"Accuracy : "
    f"{accuracy_score(y_val, val_hierarchical_encoded):.4f}"
)

print(
    f"Macro F1 : "
    f"{f1_score(y_val, val_hierarchical_encoded, average='macro'):.4f}"
)


# ============================================================
# HIERARCHICAL TEST
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL TEST")
print("=" * 70)

test_hierarchical_pred = hierarchical_predict(
    X_test
)

test_hierarchical_encoded = label_encoder.transform(
    test_hierarchical_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_hierarchical_encoded
)

test_macro_f1 = f1_score(
    y_test,
    test_hierarchical_encoded,
    average="macro"
)


print(
    f"Accuracy : {test_accuracy:.4f}"
)

print(
    f"Macro F1 : {test_macro_f1:.4f}"
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    family_svm,
    f"{MODEL_DIR}/hierarchical_family_svm.pkl"
)

joblib.dump(
    family_models,
    f"{MODEL_DIR}/hierarchical_family_models.pkl"
)

joblib.dump(
    family_encoder,
    f"{MODEL_DIR}/family_encoder.pkl"
)


np.save(
    f"{DATA_DIR}/hierarchical_test_predictions.npy",
    test_hierarchical_encoded
)


print("\nModels saved successfully.")

print("\n" + "=" * 70)
print("HIERARCHICAL SVM COMPLETED")
print("=" * 70)