import numpy as np
import os


# --------------------------------------------------
# 1. Load SVM training predictions
# --------------------------------------------------

# We will generate these predictions from the saved SVM.
from joblib import load

svm_model = load("models/svm_classifier.pkl")

X_train_latent = np.load(
    "data/processed/X_train_latent.npy"
)

X_val_latent = np.load(
    "data/processed/X_val_latent.npy"
)

X_test_latent = np.load(
    "data/processed/X_test_latent.npy"
)


print("Latent feature shapes:")
print("Train:", X_train_latent.shape)
print("Validation:", X_val_latent.shape)
print("Test:", X_test_latent.shape)


# --------------------------------------------------
# 2. Generate SVM predictions
# --------------------------------------------------

print("\nGenerating SVM predictions...")

train_predictions = svm_model.predict(X_train_latent)
val_predictions = svm_model.predict(X_val_latent)
test_predictions = svm_model.predict(X_test_latent)


print("Train predictions:", train_predictions.shape)
print("Validation predictions:", val_predictions.shape)
print("Test predictions:", test_predictions.shape)


# --------------------------------------------------
# 3. Create fixed-length sequences
# --------------------------------------------------

SEQUENCE_LENGTH = 10


def create_sequences(predictions, sequence_length):
    """
    Convert a 1D sequence of predicted attack classes
    into fixed-length observation sequences.
    """

    sequences = []

    for i in range(
        0,
        len(predictions) - sequence_length + 1,
        sequence_length
    ):
        sequence = predictions[
            i:i + sequence_length
        ]

        sequences.append(sequence)

    return np.array(sequences)


# --------------------------------------------------
# 4. Create sequences for each dataset split
# --------------------------------------------------

train_sequences = create_sequences(
    train_predictions,
    SEQUENCE_LENGTH
)

val_sequences = create_sequences(
    val_predictions,
    SEQUENCE_LENGTH
)

test_sequences = create_sequences(
    test_predictions,
    SEQUENCE_LENGTH
)


# --------------------------------------------------
# 5. Save sequences
# --------------------------------------------------

os.makedirs(
    "data/processed/hmm_sequences",
    exist_ok=True
)


np.save(
    "data/processed/hmm_sequences/train_sequences.npy",
    train_sequences
)

np.save(
    "data/processed/hmm_sequences/val_sequences.npy",
    val_sequences
)

np.save(
    "data/processed/hmm_sequences/test_sequences.npy",
    test_sequences
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nHMM sequence preparation completed.")

print("\nSequence shapes:")
print("Train sequences:", train_sequences.shape)
print("Validation sequences:", val_sequences.shape)
print("Test sequences:", test_sequences.shape)

print("\nSequence length:", SEQUENCE_LENGTH)

print("\nExample training sequence:")
print(train_sequences[0])