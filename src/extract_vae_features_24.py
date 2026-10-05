import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

ENCODER_PATH = f"{MODEL_DIR}/vae_beta_encoder_24.keras"


# ============================================================
# CUSTOM SAMPLING LAYER
# ============================================================

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(z_mean)
        )

        return (
            z_mean
            + keras.backend.exp(
                0.5 * z_log_var
            ) * epsilon
        )


# ============================================================
# LOAD ENCODER
# ============================================================

print("Loading β-VAE encoder...")

encoder = keras.models.load_model(
    ENCODER_PATH,
    custom_objects={
        "Sampling": Sampling
    },
    compile=False
)

print("Encoder loaded successfully.")


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
).astype("float32")

X_val = np.load(
    f"{DATA_DIR}/X_val.npy"
).astype("float32")

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
).astype("float32")


print("\nOriginal feature shapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ============================================================
# EXTRACT LATENT FEATURES
# ============================================================

print("\nExtracting β-VAE latent features...")


train_outputs = encoder.predict(
    X_train,
    batch_size=512,
    verbose=1
)

val_outputs = encoder.predict(
    X_val,
    batch_size=512,
    verbose=1
)

test_outputs = encoder.predict(
    X_test,
    batch_size=512,
    verbose=1
)


# ============================================================
# USE z_mean
# ============================================================
#
# Encoder outputs:
#   [0] = z_mean
#   [1] = z_log_var
#   [2] = sampled z
#
# We use z_mean because it is deterministic.
# ============================================================

X_train_latent = train_outputs[0]
X_val_latent = val_outputs[0]
X_test_latent = test_outputs[0]


# ============================================================
# SAVE LATENT FEATURES
# ============================================================

train_path = (
    f"{DATA_DIR}/X_train_vae_beta_24.npy"
)

val_path = (
    f"{DATA_DIR}/X_val_vae_beta_24.npy"
)

test_path = (
    f"{DATA_DIR}/X_test_vae_beta_24.npy"
)


np.save(
    train_path,
    X_train_latent
)

np.save(
    val_path,
    X_val_latent
)

np.save(
    test_path,
    X_test_latent
)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 60)
print("β-VAE FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print(
    "X_train VAE latent:",
    X_train_latent.shape
)

print(
    "X_val VAE latent  :",
    X_val_latent.shape
)

print(
    "X_test VAE latent :",
    X_test_latent.shape
)

print("\nSaved files:")
print(train_path)
print(val_path)
print(test_path)

print("=" * 60)