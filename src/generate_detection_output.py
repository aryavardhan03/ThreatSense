import numpy as np
import joblib
import json
import os


# ============================================================
# 1. LOAD BEST SVM
# ============================================================

svm_model = joblib.load(
    "models/svm_tuned_extended.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)


# ============================================================
# 2. LOAD ORIGINAL TEST FEATURES
# ============================================================

X_test = np.load(
    "data/processed/X_test.npy"
)

test_sequences = np.load(
    "data/processed/hmm_sequences/test_sequences.npy"
)

test_hidden_states = np.load(
    "data/processed/hmm_sequences/test_hidden_states.npy"
)


# ============================================================
# 3. GENERATE SVM PREDICTIONS
# ============================================================

print("Generating predictions using best SVM...")

predicted_classes = svm_model.predict(
    X_test
)

prediction_probabilities = svm_model.predict_proba(
    X_test
)

prediction_confidence = np.max(
    prediction_probabilities,
    axis=1
)


# ============================================================
# 4. CREATE SEQUENCE-LEVEL RECORDS
# ============================================================

detection_records = []

sequence_length = test_sequences.shape[1]

number_of_sequences = (
    len(test_sequences)
)


for sequence_index in range(
    number_of_sequences
):

    start_index = (
        sequence_index
        * sequence_length
    )

    end_index = (
        start_index
        + sequence_length
    )


    # --------------------------------------------------------
    # SVM predictions for this window
    # --------------------------------------------------------

    sequence_predictions = (
        predicted_classes[
            start_index:end_index
        ]
    )

    sequence_confidences = (
        prediction_confidence[
            start_index:end_index
        ]
    )


    attack_names = (
        label_encoder.inverse_transform(
            sequence_predictions
        )
    )


    # --------------------------------------------------------
    # HMM hidden states
    # --------------------------------------------------------

    hidden_states = (
        test_hidden_states[
            sequence_index
        ]
        .tolist()
    )


    # ========================================================
    # SEQUENCE ANALYSIS
    # ========================================================

    unique_attacks, attack_counts = (
        np.unique(
            attack_names,
            return_counts=True
        )
    )


    dominant_attack_index = (
        np.argmax(attack_counts)
    )

    dominant_attack = (
        unique_attacks[
            dominant_attack_index
        ]
    )


    number_of_unique_attacks = (
        len(unique_attacks)
    )


    # --------------------------------------------------------
    # Attack transitions
    # --------------------------------------------------------

    attack_transitions = []

    for i in range(
        len(attack_names) - 1
    ):

        source = attack_names[i]
        target = attack_names[i + 1]

        if source != target:

            attack_transitions.append({
                "from": source,
                "to": target
            })


    # --------------------------------------------------------
    # HMM transitions
    # --------------------------------------------------------

    hmm_transitions = []

    for i in range(
        len(hidden_states) - 1
    ):

        source = hidden_states[i]
        target = hidden_states[i + 1]

        if source != target:

            hmm_transitions.append({
                "from": int(source),
                "to": int(target)
            })


    # --------------------------------------------------------
    # Number of HMM state changes
    # --------------------------------------------------------

    hmm_state_changes = sum(
        1
        for i in range(
            len(hidden_states) - 1
        )
        if hidden_states[i]
        != hidden_states[i + 1]
    )


    # --------------------------------------------------------
    # Number of attack changes
    # --------------------------------------------------------

    attack_changes = sum(
        1
        for i in range(
            len(attack_names) - 1
        )
        if attack_names[i]
        != attack_names[i + 1]
    )


    # ========================================================
    # CREATE RECORD
    # ========================================================

    record = {

        "sequence_id":
            sequence_index + 1,

        "sequence_length":
            int(sequence_length),

        "predicted_attacks":
            attack_names.tolist(),

        "prediction_confidence": [
            round(
                float(confidence),
                4
            )
            for confidence
            in sequence_confidences
        ],

        "average_confidence":
            round(
                float(
                    np.mean(
                        sequence_confidences
                    )
                ),
                4
            ),

        "dominant_attack":
            dominant_attack,

        "unique_attack_count":
            int(
                number_of_unique_attacks
            ),

        "attack_transitions":
            attack_transitions,

        "attack_change_count":
            int(
                attack_changes
            ),

        "hmm_hidden_states":
            hidden_states,

        "hmm_transitions":
            hmm_transitions,

        "hmm_state_change_count":
            int(
                hmm_state_changes
            )
    }


    detection_records.append(
        record
    )


# ============================================================
# 5. SAVE OUTPUT
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

output_path = (
    "results/ml_detection_output.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        detection_records,
        file,
        indent=4
    )


# ============================================================
# 6. DISPLAY
# ============================================================

print(
    "\nML detection output generated."
)

print(
    "Total sequences:",
    len(detection_records)
)

print(
    "\nExample detection:"
)

print(
    json.dumps(
        detection_records[0],
        indent=4
    )
)

print(
    "\nSaved to:"
)

print(
    output_path
)