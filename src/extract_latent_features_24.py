import numpy as np
from tensorflow.keras.models import load_model

# Load encoder
encoder = load_model("models/encoder_24.keras")

# Load data
X_train = np.load("data/processed/X_train.npy")
X_val = np.load("data/processed/X_val.npy")
X_test = np.load("data/processed/X_test.npy")

print("Original shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# Extract latent features
X_train_latent = encoder.predict(X_train, batch_size=512)
X_val_latent = encoder.predict(X_val, batch_size=512)
X_test_latent = encoder.predict(X_test, batch_size=512)

print("\nLatent shapes:")
print("X_train_latent:", X_train_latent.shape)
print("X_val_latent:", X_val_latent.shape)
print("X_test_latent:", X_test_latent.shape)

# Save
np.save(
    "data/processed/X_train_latent_24.npy",
    X_train_latent
)

np.save(
    "data/processed/X_val_latent_24.npy",
    X_val_latent
)

np.save(
    "data/processed/X_test_latent_24.npy",
    X_test_latent
)

print("\n24-D latent features saved successfully.")