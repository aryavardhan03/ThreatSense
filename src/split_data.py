import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/ciciot_5000_per_class.csv"
OUTPUT_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CICIoT2023 TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURES / LABEL
# ============================================================

X = df.drop(
    columns=["Label"]
)

y = df["Label"]


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print(
    f"Number of classes: "
    f"{len(label_encoder.classes_)}"
)

print("\nClass mapping:")

for i, name in enumerate(
    label_encoder.classes_
):
    print(
        f"{i:2d} -> {name}"
    )


# ============================================================
# 70 / 15 / 15 STRATIFIED SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y_encoded,
        test_size=0.30,
        stratify=y_encoded,
        random_state=42
    )
)

X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )
)


print("\nSplit sizes:")

print(
    f"Train      : {X_train.shape}"
)

print(
    f"Validation : {X_val.shape}"
)

print(
    f"Test       : {X_test.shape}"
)


# ============================================================
# STANDARD SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# SAVE ARRAYS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train_scaled
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_val.npy"
    ),
    X_val_scaled
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test_scaled
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_val.npy"
    ),
    y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)


# ============================================================
# SAVE SCALER + LABEL ENCODER
# ============================================================

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler_5000.pkl"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        MODEL_DIR,
        "label_encoder_5000.pkl"
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SPLIT COMPLETED")
print("=" * 70)

print(
    f"X_train: {X_train_scaled.shape}"
)

print(
    f"X_val  : {X_val_scaled.shape}"
)

print(
    f"X_test : {X_test_scaled.shape}"
)

print("\nSaved:")
print("  X_train.npy")
print("  X_val.npy")
print("  X_test.npy")
print("  y_train.npy")
print("  y_val.npy")
print("  y_test.npy")
print("  scaler_5000.pkl")
print("  label_encoder_5000.pkl")

print("=" * 70)