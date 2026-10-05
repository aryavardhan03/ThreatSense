import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os


# --------------------------------------------------
# 1. Load scaled training and validation data
# --------------------------------------------------

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)


# --------------------------------------------------
# 2. Build Autoencoder
# --------------------------------------------------

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)

# Latent representation
latent = Dense(16, activation="relu", name="latent_features")(encoded)

# Decoder
decoded = Dense(32, activation="relu")(latent)
decoded = Dense(64, activation="relu")(decoded)
output_layer = Dense(input_dim, activation="linear")(decoded)


# Complete Autoencoder
autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)


# --------------------------------------------------
# 3. Compile Autoencoder
# --------------------------------------------------

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


# --------------------------------------------------
# 4. Display architecture
# --------------------------------------------------

print("\nAutoencoder architecture:")
autoencoder.summary()


# --------------------------------------------------
# 5. Early stopping
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# --------------------------------------------------
# 6. Train Autoencoder
# --------------------------------------------------

print("\nStarting Autoencoder training...")

history = autoencoder.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=30,
    batch_size=512,
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------------------------------
# 7. Create models directory
# --------------------------------------------------

os.makedirs("models", exist_ok=True)


# --------------------------------------------------
# 8. Save trained Autoencoder
# --------------------------------------------------

autoencoder.save(
    "models/autoencoder.keras"
)


# --------------------------------------------------
# 9. Extract encoder model
# --------------------------------------------------

encoder = Model(
    inputs=input_layer,
    outputs=latent
)


encoder.save(
    "models/encoder.keras"
)


# --------------------------------------------------
# 10. Final information
# --------------------------------------------------

print("\nAutoencoder training completed successfully.")

print("Autoencoder saved to:")
print("models/autoencoder.keras")

print("\nEncoder saved to:")
print("models/encoder.keras")

print("\nLatent feature dimension:", 16)