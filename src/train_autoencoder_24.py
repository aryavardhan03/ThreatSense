import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Load data
# -----------------------------

X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

# -----------------------------
# Autoencoder architecture
# -----------------------------

input_dim = X_train.shape[1]
latent_dim = 24

inputs = Input(shape=(input_dim,))

# Encoder
x = Dense(64, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
latent = Dense(latent_dim, activation="relu", name="latent_features")(x)

# Decoder
x = Dense(32, activation="relu")(latent)
x = Dense(64, activation="relu")(x)
outputs = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs, outputs)

# Separate encoder
encoder = Model(inputs, latent)

# -----------------------------
# Compile
# -----------------------------

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

# -----------------------------
# Early stopping
# -----------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Train
# -----------------------------

history = autoencoder.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=30,
    batch_size=512,
    callbacks=[early_stopping],
    verbose=1
)

# -----------------------------
# Save models
# -----------------------------

autoencoder.save(
    "models/autoencoder_24.keras"
)

encoder.save(
    "models/encoder_24.keras"
)

print("\n24-dimensional Autoencoder training completed.")

print("Saved:")
print("models/autoencoder_24.keras")
print("models/encoder_24.keras")

print("\nFinal training loss:",
      history.history["loss"][-1])

print("Final validation loss:",
      history.history["val_loss"][-1])