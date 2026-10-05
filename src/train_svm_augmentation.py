import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors


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
# FEATURE SELECTION
# ============================================================

selector = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_selector.pkl"
    )
)

X_train = selector.transform(X_train)
X_val = selector.transform(X_val)


print("=" * 75)
print("CLASS-AWARE DATA AUGMENTATION + SVM")
print("=" * 75)

print(
    f"Training shape   : {X_train.shape}"
)

print(
    f"Validation shape : {X_val.shape}"
)


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


# ============================================================
# DIFFICULT CLASSES
# ============================================================

difficult_classes = [
    "RECON-OSSCAN",
    "XSS",
    "VULNERABILITYSCAN",
    "RECON-PINGSWEEP",
    "RECON-PORTSCAN",
    "SQLINJECTION",
    "UPLOADING_ATTACK",
    "BACKDOOR_MALWARE",
    "DDOS-SYN_FLOOD"
]


difficult_labels = []

for class_name in difficult_classes:

    if class_name in class_names:

        label = np.where(
            class_names == class_name
        )[0][0]

        difficult_labels.append(label)


print("\nDifficult classes:")

for label in difficult_labels:
    print(
        f"  {label}: {class_names[label]}"
    )


# ============================================================
# SYNTHETIC SAMPLE GENERATION
# ============================================================

def generate_synthetic_samples(
    X,
    y,
    target_label,
    multiplier,
    random_state=42
):

    rng = np.random.default_rng(
        random_state
    )

    class_indices = np.where(
        y == target_label
    )[0]

    X_class = X[class_indices]

    if len(X_class) < 3:
        return np.empty(
            (0, X.shape[1])
        )

    # Number of synthetic samples
    n_new = int(
        len(X_class) * multiplier
    )

    # Nearest neighbors inside the same class
    n_neighbors = min(
        5,
        len(X_class) - 1
    )

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1
    )

    nn.fit(X_class)

    distances, neighbors = nn.kneighbors(
        X_class
    )

    synthetic = []

    for _ in range(n_new):

        # Select random base sample
        base_index = rng.integers(
            0,
            len(X_class)
        )

        base = X_class[base_index]

        # Select one of its neighbours
        neighbor_choices = neighbors[
            base_index
        ][1:]

        neighbor_index = rng.choice(
            neighbor_choices
        )

        neighbor = X_class[
            neighbor_index
        ]

        # Interpolation factor
        alpha = rng.random()

        synthetic_sample = (
            base
            + alpha
            * (neighbor - base)
        )

        synthetic.append(
            synthetic_sample
        )

    return np.asarray(
        synthetic
    )


# ============================================================
# AUGMENTATION LEVELS
# ============================================================

augmentation_levels = [
    0.25,
    0.50,
    1.00,
    1.50
]


results = []

best_macro_f1 = -1
best_accuracy = 0
best_level = None
best_model = None


# ============================================================
# EXPERIMENTS
# ============================================================

for augmentation_level in augmentation_levels:

    print("\n" + "=" * 75)

    print(
        f"Augmentation multiplier: "
        f"{augmentation_level}"
    )

    print("=" * 75)


    synthetic_X = []
    synthetic_y = []


    # --------------------------------------------------------
    # Generate synthetic samples
    # --------------------------------------------------------

    for label in difficult_labels:

        X_new = generate_synthetic_samples(
            X_train,
            y_train,
            label,
            augmentation_level,
            random_state=42 + label
        )

        if len(X_new) > 0:

            synthetic_X.append(
                X_new
            )

            synthetic_y.extend(
                [label] * len(X_new)
            )

            print(
                f"{class_names[label]:<30}"
                f"+{len(X_new)} samples"
            )


    # --------------------------------------------------------
    # Combine original + synthetic
    # --------------------------------------------------------

    if synthetic_X:

        synthetic_X = np.vstack(
            synthetic_X
        )

        synthetic_y = np.asarray(
            synthetic_y
        )

        X_augmented = np.vstack(
            [
                X_train,
                synthetic_X
            ]
        )

        y_augmented = np.concatenate(
            [
                y_train,
                synthetic_y
            ]
        )

    else:

        X_augmented = X_train
        y_augmented = y_train


    print(
        f"\nOriginal samples : "
        f"{len(X_train)}"
    )

    print(
        f"Synthetic samples: "
        f"{len(synthetic_X)}"
    )

    print(
        f"Final samples    : "
        f"{len(X_augmented)}"
    )


    # --------------------------------------------------------
    # Train SVM
    # --------------------------------------------------------

    svm = SVC(
        kernel="rbf",
        C=500,
        gamma=0.02,
        class_weight=None,
        probability=True,
        random_state=42
    )

    print("\nTraining SVM...")

    svm.fit(
        X_augmented,
        y_augmented
    )

    # --------------------------------------------------------
    # Validation
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
        "augmentation": augmentation_level,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })


    # --------------------------------------------------------
    # Track best
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_accuracy = accuracy
        best_level = augmentation_level
        best_model = svm


# ============================================================
# RESULTS TABLE
# ============================================================

print("\n\n")

print("=" * 75)
print("AUGMENTATION RESULTS")
print("=" * 75)

print(
    f"{'Multiplier':<15}"
    f"{'Accuracy':<15}"
    f"{'Macro F1':<15}"
    f"{'Weighted F1':<15}"
)

print("-" * 75)

for result in results:

    print(
        f"{result['augmentation']:<15}"
        f"{result['accuracy']:<15.4f}"
        f"{result['macro_f1']:<15.4f}"
        f"{result['weighted_f1']:<15.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

print("\n")

print("=" * 75)
print("BEST AUGMENTED SVM")
print("=" * 75)

print(
    f"Augmentation multiplier : "
    f"{best_level}"
)

print(
    f"Validation Accuracy     : "
    f"{best_accuracy:.4f}"
)

print(
    f"Validation Macro F1     : "
    f"{best_macro_f1:.4f}"
)


# ============================================================
# SAVE
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "svm_augmented.pkl"
)

joblib.dump(
    best_model,
    model_path
)

results_path = os.path.join(
    MODEL_DIR,
    "svm_augmentation_results.pkl"
)

joblib.dump(
    results,
    results_path
)

print("\nSaved:")
print(model_path)
print(results_path)

print("\nClass-aware augmentation experiment completed.")