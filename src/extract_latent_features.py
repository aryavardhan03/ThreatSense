import numpy as np
from tensorflow.keras.models import load_model
import os


# --------------------------------------------------
# 1. Load scaled datasets
# --------------------------------------------------

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
X_test = np.load("data/processed/X_test.npy")

print("Original feature shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# --------------------------------------------------
# 2. Load trained encoder
# --------------------------------------------------

encoder = load_model("models/encoder.keras")

print("\nEncoder loaded successfully.")


# --------------------------------------------------
# 3. Extract latent features
# --------------------------------------------------

X_train_latent = encoder.predict(
    X_train,
    batch_size=512,
    verbose=1
)

X_val_latent = encoder.predict(
    X_val,
    batch_size=512,
    verbose=1
)

X_test_latent = encoder.predict(
    X_test,
    batch_size=512,
    verbose=1
)


# --------------------------------------------------
# 4. Create output directory
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)


# --------------------------------------------------
# 5. Save latent features
# --------------------------------------------------

np.save(
    "data/processed/X_train_latent.npy",
    X_train_latent
)

np.save(
    "data/processed/X_val_latent.npy",
    X_val_latent
)

np.save(
    "data/processed/X_test_latent.npy",
    X_test_latent
)


# --------------------------------------------------
# 6. Display final shapes
# --------------------------------------------------

print("\nLatent feature extraction completed.")

print("\nLatent feature shapes:")
print("X_train_latent:", X_train_latent.shape)
print("X_val_latent:", X_val_latent.shape)
print("X_test_latent:", X_test_latent.shape)

print("\nLatent features saved successfully.")