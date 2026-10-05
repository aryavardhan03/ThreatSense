import numpy as np
import joblib
import os

from hmmlearn.hmm import CategoricalHMM


# --------------------------------------------------
# 1. Load HMM sequences
# --------------------------------------------------

TRAIN_PATH = "data/processed/hmm_sequences/train_sequences.npy"
VAL_PATH = "data/processed/hmm_sequences/val_sequences.npy"
TEST_PATH = "data/processed/hmm_sequences/test_sequences.npy"

train_sequences = np.load(TRAIN_PATH)
val_sequences = np.load(VAL_PATH)
test_sequences = np.load(TEST_PATH)


print("HMM sequence shapes:")
print("Train:", train_sequences.shape)
print("Validation:", val_sequences.shape)
print("Test:", test_sequences.shape)


# --------------------------------------------------
# 2. Convert sequences to integer observations
# --------------------------------------------------

# Each SVM prediction is already an integer class ID.
# CategoricalHMM expects observations in the form:
#
# [[state],
#  [state],
#  [state],
#  ...]
#
# Therefore flatten the sequences.

X_train_hmm = train_sequences.reshape(-1, 1)

X_val_hmm = val_sequences.reshape(-1, 1)

X_test_hmm = test_sequences.reshape(-1, 1)


# Length of each sequence
train_lengths = [train_sequences.shape[1]] * train_sequences.shape[0]
val_lengths = [val_sequences.shape[1]] * val_sequences.shape[0]
test_lengths = [test_sequences.shape[1]] * test_sequences.shape[0]


print("\nHMM observation shapes:")
print("Training observations:", X_train_hmm.shape)
print("Validation observations:", X_val_hmm.shape)
print("Test observations:", X_test_hmm.shape)


# --------------------------------------------------
# 3. Define HMM
# --------------------------------------------------

NUMBER_OF_HIDDEN_STATES = 10
NUMBER_OF_OBSERVATIONS = 34


hmm_model = CategoricalHMM(
    n_components=NUMBER_OF_HIDDEN_STATES,
    n_iter=100,
    tol=0.001,
    random_state=42,
    verbose=True
)

# Explicitly tell the model that there are
# 34 possible SVM attack-class observations.

hmm_model.n_features = NUMBER_OF_OBSERVATIONS


# --------------------------------------------------
# 4. Train HMM
# --------------------------------------------------

print("\nStarting HMM training...")

hmm_model.fit(
    X_train_hmm,
    lengths=train_lengths
)


print("\nHMM training completed successfully.")


# --------------------------------------------------
# 5. Calculate validation log likelihood
# --------------------------------------------------

val_score = hmm_model.score(
    X_val_hmm,
    lengths=val_lengths
)

print("\nValidation log likelihood:")
print(val_score)


# --------------------------------------------------
# 6. Calculate test log likelihood
# --------------------------------------------------

test_score = hmm_model.score(
    X_test_hmm,
    lengths=test_lengths
)

print("\nTest log likelihood:")
print(test_score)


# --------------------------------------------------
# 7. Decode hidden states
# --------------------------------------------------

print("\nDecoding test sequences...")

hidden_states = []

for sequence in test_sequences:

    observations = sequence.reshape(-1, 1)

    states = hmm_model.predict(observations)

    hidden_states.append(states)


hidden_states = np.array(hidden_states)


print("Hidden state shape:", hidden_states.shape)


# --------------------------------------------------
# 8. Display transition matrix
# --------------------------------------------------

print("\nHMM Transition Matrix:")

print(hmm_model.transmat_)


# --------------------------------------------------
# 9. Save hidden states
# --------------------------------------------------

os.makedirs(
    "data/processed/hmm_sequences",
    exist_ok=True
)

np.save(
    "data/processed/hmm_sequences/test_hidden_states.npy",
    hidden_states
)


# --------------------------------------------------
# 10. Save HMM model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    hmm_model,
    "models/hmm_model.pkl"
)


print("\nHMM model saved:")
print("models/hmm_model.pkl")

print("\nHidden states saved:")
print(
    "data/processed/hmm_sequences/test_hidden_states.npy"
)