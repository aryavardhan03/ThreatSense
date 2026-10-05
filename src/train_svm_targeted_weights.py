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

selector = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_selector.pkl"
    )
)

X_train = selector.transform(X_train)
X_val = selector.transform(X_val)


# ============================================================
# LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)

class_names = label_encoder.classes_

print("=" * 75)
print("TARGETED CLASS-WEIGHT SVM EXPERIMENT")
print("=" * 75)

print("Training shape :", X_train.shape)
print("Validation shape:", X_val.shape)


# ============================================================
# DIFFICULT CLASSES
# ============================================================

difficult_classes = [
    "RECON-OSSCAN",
    "XSS",
    "VULNERABILITYSCAN",
    "RECON-PINGSWEEP",
    "SQLINJECTION",
    "UPLOADING_ATTACK",
    "BACKDOOR_MALWARE",
    "DDOS-SYN_FLOOD"
]


# ============================================================
# WEIGHT LEVELS
# ============================================================

weight_levels = [
    1.25,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0
]


# ============================================================
# BASE PARAMETERS
# ============================================================

C = 500
GAMMA = 0.02


results = []

best_macro_f1 = -1
best_accuracy = 0
best_weight = None
best_model = None


# ============================================================
# EXPERIMENT
# ============================================================

for weight_multiplier in weight_levels:

    print("\n" + "=" * 75)
    print(
        f"Testing difficult-class weight = "
        f"{weight_multiplier}"
    )
    print("=" * 75)

    # Start every class with weight 1
    class_weights = {
        i: 1.0
        for i in range(len(class_names))
    }

    # Increase weight of difficult classes
    for class_name in difficult_classes:

        if class_name in class_names:

            class_index = np.where(
                class_names == class_name
            )[0][0]

            class_weights[class_index] = weight_multiplier

    print("\nApplied weights:")

    for class_name in difficult_classes:

        if class_name in class_names:

            class_index = np.where(
                class_names == class_name
            )[0][0]

            print(
                f"{class_name:<30} "
                f"{class_weights[class_index]}"
            )

    # --------------------------------------------------------
    # Train SVM
    # --------------------------------------------------------

    svm = SVC(
        kernel="rbf",
        C=C,
        gamma=GAMMA,
        class_weight=class_weights,
        probability=True,
        random_state=42
    )

    svm.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    y_pred = svm.predict(
        X_val
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

    print("\nResults:")
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
        "weight": weight_multiplier,
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
        best_weight = weight_multiplier
        best_model = svm


# ============================================================
# RESULTS
# ============================================================

print("\n\n")
print("=" * 75)
print("TARGETED WEIGHT RESULTS")
print("=" * 75)

print(
    f"{'Weight':<12}"
    f"{'Accuracy':<15}"
    f"{'Macro F1':<15}"
    f"{'Weighted F1':<15}"
)

print("-" * 75)

for result in results:

    print(
        f"{result['weight']:<12}"
        f"{result['accuracy']:<15.4f}"
        f"{result['macro_f1']:<15.4f}"
        f"{result['weighted_f1']:<15.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

print("\n")
print("=" * 75)
print("BEST TARGETED-WEIGHT MODEL")
print("=" * 75)

print(
    f"Weight multiplier : {best_weight}"
)

print(
    f"Validation Accuracy : {best_accuracy:.4f}"
)

print(
    f"Validation Macro F1 : {best_macro_f1:.4f}"
)


# ============================================================
# SAVE
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "svm_targeted_weights.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nSaved:")
print(model_path)

print("\nExperiment completed.")