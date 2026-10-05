import json
import os


# --------------------------------------------------
# 1. Load official MITRE ATT&CK STIX data
# --------------------------------------------------

INPUT_PATH = "data/cti/enterprise-attack.json"

print("Loading MITRE ATT&CK data...")

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as file:
    attack_data = json.load(file)


objects = attack_data["objects"]

print("Total STIX objects:", len(objects))


# --------------------------------------------------
# 2. Extract Enterprise ATT&CK techniques
# --------------------------------------------------

techniques = []

for obj in objects:

    # We only want Enterprise ATT&CK attack-patterns
    if obj.get("type") != "attack-pattern":
        continue

    # Ignore revoked/deprecated techniques
    if obj.get("revoked", False):
        continue

    if obj.get("x_mitre_deprecated", False):
        continue

    external_references = obj.get(
        "external_references",
        []
    )

    technique_id = None

    for reference in external_references:

        if (
            reference.get("source_name")
            == "mitre-attack"
        ):
            technique_id = reference.get(
                "external_id"
            )
            break

    if technique_id is None:
        continue


    # --------------------------------------------------
    # 3. Extract tactic information
    # --------------------------------------------------

    kill_chain_phases = obj.get(
        "kill_chain_phases",
        []
    )

    tactics = []

    for phase in kill_chain_phases:

        if (
            phase.get("kill_chain_name")
            == "mitre-attack"
        ):
            tactics.append(
                phase.get("phase_name")
            )


    # --------------------------------------------------
    # 4. Create simplified technique record
    # --------------------------------------------------

    technique = {
        "technique_id": technique_id,

        "name": obj.get(
            "name",
            ""
        ),

        "description": obj.get(
            "description",
            ""
        ),

        "tactics": tactics,

        "stix_id": obj.get(
            "id",
            ""
        )
    }

    techniques.append(
        technique
    )


# --------------------------------------------------
# 5. Sort techniques by ATT&CK ID
# --------------------------------------------------

techniques.sort(
    key=lambda x: x["technique_id"]
)


# --------------------------------------------------
# 6. Save simplified CTI knowledge base
# --------------------------------------------------

OUTPUT_PATH = (
    "data/cti/mitre_techniques.json"
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        techniques,
        file,
        indent=4,
        ensure_ascii=False
    )


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\nMITRE ATT&CK parsing completed.")

print(
    "Enterprise techniques extracted:",
    len(techniques)
)

print("\nExample techniques:")

for technique in techniques[:5]:

    print(
        technique["technique_id"],
        "->",
        technique["name"]
    )


print("\nSaved to:")
print(OUTPUT_PATH)