import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

print("=" * 70)
print("SVM FEATURE SELECTION EXPERIMENT")
print("=" * 70)

print(f"Training data : {X_train.shape}")
print(f"Validation data: {X_val.shape}")


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance"
]

print(f"\nTotal features: {len(feature_names)}")


# ============================================================
# FEATURE SELECTION
# ============================================================

K_VALUES = [10, 15, 20, 25, 30, 35, 39]

results = []

best_macro_f1 = -1
best_k = None
best_selector = None
best_model = None


# ============================================================
# EXPERIMENT
# ============================================================

for k in K_VALUES:

    print("\n" + "=" * 70)
    print(f"Testing Top {k} Features")
    print("=" * 70)

    # Select features using training data only
    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm = SVC(
        kernel="rbf",
        C=500,
        gamma=0.02,
        class_weight=None,
        probability=True,
        random_state=42
    )

    svm.fit(X_train_selected, y_train)

    y_pred = svm.predict(X_val_selected)

    accuracy = accuracy_score(y_val, y_pred)

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted"
    )

    print(f"Features      : {k}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")

    # --------------------------------------------------------
    # Selected feature names
    # --------------------------------------------------------

    selected_mask = selector.get_support()

    selected_features = [
        name
        for name, selected in zip(feature_names, selected_mask)
        if selected
    ]

    print("\nSelected features:")

    for feature in selected_features:
        print(f"  - {feature}")

    results.append({
        "k": k,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "features": selected_features
    })

    # --------------------------------------------------------
    # Track best model using Macro F1
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_k = k
        best_selector = selector
        best_model = svm


# ============================================================
# SUMMARY
# ============================================================

print("\n\n")
print("=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    f"{'Features':<12}"
    f"{'Accuracy':<15}"
    f"{'Macro F1':<15}"
    f"{'Weighted F1':<15}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['k']:<12}"
        f"{result['accuracy']:<15.4f}"
        f"{result['macro_f1']:<15.4f}"
        f"{result['weighted_f1']:<15.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST FEATURE SET")
print("=" * 70)

print(f"Best number of features : {best_k}")
print(f"Best validation Macro F1: {best_macro_f1:.4f}")

best_features = [
    name
    for name, selected in zip(
        feature_names,
        best_selector.get_support()
    )
    if selected
]

print("\nBest selected features:")

for feature in best_features:
    print(f"  - {feature}")


# ============================================================
# SAVE SELECTOR + MODEL
# ============================================================

joblib.dump(
    best_selector,
    os.path.join(MODEL_DIR, "feature_selector.pkl")
)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "svm_feature_selected.pkl")
)

print("\nSaved:")
print("models/feature_selector.pkl")
print("models/svm_feature_selected.pkl")

print("\nFeature selection experiment completed.")