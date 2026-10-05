import json
import os


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DETECTION_PATH = (
    "results/ml_detection_output.json"
)

MAPPING_PATH = (
    "data/cti/attack_mapping.json"
)

MITRE_PATH = (
    "data/cti/mitre_techniques.json"
)

OUTPUT_PATH = (
    "results/cti_enriched_output.json"
)


# --------------------------------------------------
# 2. Load files
# --------------------------------------------------

print("Loading ML detection output...")

with open(
    DETECTION_PATH,
    "r",
    encoding="utf-8"
) as file:
    detection_records = json.load(file)


print("Loading attack mapping...")

with open(
    MAPPING_PATH,
    "r",
    encoding="utf-8"
) as file:
    attack_mapping = json.load(file)


print("Loading MITRE ATT&CK knowledge base...")

with open(
    MITRE_PATH,
    "r",
    encoding="utf-8"
) as file:
    mitre_techniques = json.load(file)


# --------------------------------------------------
# 3. Create MITRE lookup dictionary
# --------------------------------------------------

mitre_lookup = {}

for technique in mitre_techniques:

    technique_id = technique[
        "technique_id"
    ]

    mitre_lookup[technique_id] = technique


print(
    "MITRE techniques available:",
    len(mitre_lookup)
)


# --------------------------------------------------
# 4. Enrich detections
# --------------------------------------------------

enriched_records = []


for record in detection_records:

    enriched_attacks = []


    for attack_name in record[
        "predicted_attacks"
    ]:

        mappings = attack_mapping.get(
            attack_name,
            []
        )

        attack_information = {
            "attack_name": attack_name,
            "mitre_mappings": []
        }


        # ------------------------------------------
        # No mapping available
        # ------------------------------------------

        if not mappings:

            attack_information[
                "mitre_mappings"
            ].append({

                "mapping_status": "unknown",

                "technique_id": None,

                "technique_name": None,

                "tactic": [],

                "description": (
                    "No explicit MITRE ATT&CK "
                    "mapping is currently defined "
                    "for this detected attack category."
                )

            })


        # ------------------------------------------
        # Mapping available
        # ------------------------------------------

        else:

            for mapping in mappings:

                technique_id = mapping[
                    "technique_id"
                ]

                mitre_information = (
                    mitre_lookup.get(
                        technique_id
                    )
                )


                # ----------------------------------
                # Verify technique exists
                # ----------------------------------

                if mitre_information:

                    enriched_mapping = {

                        "mapping_status":
                            mapping.get(
                                "mapping_status",
                                "potential"
                            ),

                        "technique_id":
                            technique_id,

                        "technique_name":
                            mitre_information[
                                "name"
                            ],

                        "tactics":
                            mitre_information[
                                "tactics"
                            ],

                        "description":
                            mitre_information[
                                "description"
                            ],

                        "mapping_basis":
                            mapping.get(
                                "mapping_basis",
                                ""
                            )
                    }

                    attack_information[
                        "mitre_mappings"
                    ].append(
                        enriched_mapping
                    )


                # ----------------------------------
                # Mapping references invalid ID
                # ----------------------------------

                else:

                    attack_information[
                        "mitre_mappings"
                    ].append({

                        "mapping_status":
                            "invalid_mapping",

                        "technique_id":
                            technique_id,

                        "technique_name":
                            None,

                        "tactics": [],

                        "description":
                            "Technique ID was not "
                            "found in the MITRE "
                            "knowledge base."

                    })


        enriched_attacks.append(
            attack_information
        )


    # --------------------------------------------------
    # 5. Create enriched record
    # --------------------------------------------------

    enriched_record = {

        "sequence_id":
            record["sequence_id"],

        "average_confidence":
            record["average_confidence"],

        "predicted_attacks":
            record["predicted_attacks"],

        "prediction_confidence":
            record["prediction_confidence"],

        "hmm_hidden_states":
            record["hmm_hidden_states"],

        "cti_enrichment":
            enriched_attacks
    }


    enriched_records.append(
        enriched_record
    )


# --------------------------------------------------
# 6. Save enriched output
# --------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        enriched_records,
        file,
        indent=4,
        ensure_ascii=False
    )


# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\nCTI enrichment completed.")

print(
    "Total sequences enriched:",
    len(enriched_records)
)

print("\nSaved to:")
print(OUTPUT_PATH)


# --------------------------------------------------
# 8. Display first sequence
# --------------------------------------------------

print("\nExample enriched sequence:")

print(
    json.dumps(
        enriched_records[0],
        indent=4
    )
)