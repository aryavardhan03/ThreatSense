import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

LATENT_DIM = 24
NOISE_FACTOR = 0.10
EPOCHS = 50
BATCH_SIZE = 512
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
)

X_val = np.load(
    f"{DATA_DIR}/X_val.npy"
)

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)

print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ============================================================
# ADD NOISE
# ============================================================

print("\nAdding Gaussian noise...")
print("Noise factor:", NOISE_FACTOR)

rng = np.random.default_rng(42)

train_noise = rng.normal(
    loc=0.0,
    scale=NOISE_FACTOR,
    size=X_train.shape
).astype(np.float32)

val_noise = rng.normal(
    loc=0.0,
    scale=NOISE_FACTOR,
    size=X_val.shape
).astype(np.float32)


X_train_noisy = X_train + train_noise
X_val_noisy = X_val + val_noise


# ============================================================
# BUILD DENOISING AUTOENCODER
# ============================================================

print("\nBuilding Denoising Autoencoder...")

# -------------------------
# Encoder
# -------------------------

encoder_input = keras.Input(
    shape=(X_train.shape[1],),
    name="encoder_input"
)

x = layers.Dense(
    64,
    activation="relu"
)(encoder_input)

x = layers.Dense(
    32,
    activation="relu"
)(x)

latent = layers.Dense(
    LATENT_DIM,
    activation="relu",
    name="latent_features"
)(x)


encoder = keras.Model(
    encoder_input,
    latent,
    name="denoising_encoder"
)


# -------------------------
# Decoder
# -------------------------

decoder_input = keras.Input(
    shape=(LATENT_DIM,),
    name="decoder_input"
)

x = layers.Dense(
    32,
    activation="relu"
)(decoder_input)

x = layers.Dense(
    64,
    activation="relu"
)(x)

decoder_output = layers.Dense(
    X_train.shape[1],
    activation="linear",
    name="reconstruction"
)(x)


decoder = keras.Model(
    decoder_input,
    decoder_output,
    name="denoising_decoder"
)


# -------------------------
# Autoencoder
# -------------------------

autoencoder_input = keras.Input(
    shape=(X_train.shape[1],),
    name="autoencoder_input"
)

encoded = encoder(
    autoencoder_input
)

decoded = decoder(
    encoded
)

autoencoder = keras.Model(
    autoencoder_input,
    decoded,
    name="denoising_autoencoder"
)


# ============================================================
# COMPILE
# ============================================================

autoencoder.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mse"
)


autoencoder.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

history = autoencoder.fit(
    X_train_noisy,
    X_train,
    validation_data=(
        X_val_noisy,
        X_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# BEST VALIDATION LOSS
# ============================================================

best_val_loss = min(
    history.history["val_loss"]
)

best_epoch = (
    np.argmin(
        history.history["val_loss"]
    ) + 1
)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Best epoch: {best_epoch}"
)


# ============================================================
# SAVE MODELS
# ============================================================

autoencoder_path = (
    f"{MODEL_DIR}/denoising_autoencoder_24.keras"
)

encoder_path = (
    f"{MODEL_DIR}/denoising_encoder_24.keras"
)

decoder_path = (
    f"{MODEL_DIR}/denoising_decoder_24.keras"
)


autoencoder.save(
    autoencoder_path
)

encoder.save(
    encoder_path
)

decoder.save(
    decoder_path
)


print("\nSaved models:")

print(autoencoder_path)
print(encoder_path)
print(decoder_path)


# ============================================================
# EXTRACT LATENT FEATURES
# ============================================================

print("\nExtracting latent features...")


X_train_latent = encoder.predict(
    X_train,
    batch_size=BATCH_SIZE,
    verbose=1
)

X_val_latent = encoder.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=1
)

X_test_latent = encoder.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# SAVE LATENT FEATURES
# ============================================================

train_latent_path = (
    f"{DATA_DIR}/X_train_denoising_latent_24.npy"
)

val_latent_path = (
    f"{DATA_DIR}/X_val_denoising_latent_24.npy"
)

test_latent_path = (
    f"{DATA_DIR}/X_test_denoising_latent_24.npy"
)


np.save(
    train_latent_path,
    X_train_latent
)

np.save(
    val_latent_path,
    X_val_latent
)

np.save(
    test_latent_path,
    X_test_latent
)


print("\nLatent feature shapes:")

print(
    "Train:",
    X_train_latent.shape
)

print(
    "Val  :",
    X_val_latent.shape
)

print(
    "Test :",
    X_test_latent.shape
)


print("\nSaved latent features:")

print(train_latent_path)
print(val_latent_path)
print(test_latent_path)


print("\n" + "=" * 70)
print("DENOISING AUTOENCODER COMPLETED")
print("=" * 70)