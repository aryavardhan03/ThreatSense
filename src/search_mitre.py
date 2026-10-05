import json


# --------------------------------------------------
# Load simplified MITRE knowledge base
# --------------------------------------------------

INPUT_PATH = "data/cti/mitre_techniques.json"

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as file:
    techniques = json.load(file)


# --------------------------------------------------
# Search function
# --------------------------------------------------

def search_techniques(keyword):

    keyword = keyword.lower()

    matches = []

    for technique in techniques:

        name = technique["name"].lower()
        description = technique["description"].lower()

        if (
            keyword in name
            or keyword in description
        ):
            matches.append(technique)

    return matches


# --------------------------------------------------
# Search selected cybersecurity concepts
# --------------------------------------------------

keywords = [
    "SQL",
    "Spearphishing",
    "Input",
    "Command",
    "Network Service Scanning",
    "Exploitation",
    "DNS",
    "Web"
]


for keyword in keywords:

    print("\n" + "=" * 60)
    print("SEARCH:", keyword)
    print("=" * 60)

    results = search_techniques(keyword)

    for technique in results[:10]:

        print(
            technique["technique_id"],
            "->",
            technique["name"]
        )

        print(
            "Tactics:",
            ", ".join(technique["tactics"])
        )

        print()