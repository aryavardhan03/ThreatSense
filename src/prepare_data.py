import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data/CICIoT2023/MERGED_CSV"
OUTPUT_DIR = "data/processed"

SAMPLES_PER_CLASS = 5000
CHUNK_SIZE = 25_000

TEMP_DIR = os.path.join(
    OUTPUT_DIR,
    "temp_5000"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ============================================================
# INITIALIZE
# ============================================================

files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
])

class_counts = {}

print("=" * 70)
print("CICIoT2023 MEMORY-SAFE DATA PREPARATION")
print("=" * 70)

print(f"CSV files found: {len(files)}")
print(f"Target samples/class: {SAMPLES_PER_CLASS}")
print(f"Chunk size: {CHUNK_SIZE}")


# ============================================================
# READ CSV FILES
# ============================================================

for file_idx, filename in enumerate(files, 1):

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    print(
        f"\n[{file_idx}/{len(files)}] "
        f"Reading {filename}"
    )

    for chunk in pd.read_csv(
        filepath,
        chunksize=CHUNK_SIZE
    ):

        chunk.columns = (
            chunk.columns
            .str.strip()
        )

        if "Label" not in chunk.columns:
            continue

        # ----------------------------------------------------
        # Work class by class
        # ----------------------------------------------------

        labels = chunk["Label"].unique()

        for label in labels:

            current = class_counts.get(
                label,
                0
            )

            if current >= SAMPLES_PER_CLASS:
                continue

            remaining = (
                SAMPLES_PER_CLASS
                - current
            )

            # Get indices first
            indices = np.flatnonzero(
                chunk["Label"].to_numpy()
                == label
            )

            if len(indices) == 0:
                continue

            indices = indices[
                :remaining
            ]

            # Extract only required rows
            selected = chunk.iloc[
                indices
            ]

            # Append directly to temporary class file
            temp_file = os.path.join(
                TEMP_DIR,
                f"{label}.csv"
            )

            write_header = not os.path.exists(
                temp_file
            )

            selected.to_csv(
                temp_file,
                mode="a",
                header=write_header,
                index=False
            )

            class_counts[label] = (
                current
                + len(selected)
            )

        del chunk

    # --------------------------------------------------------
    # Check if all classes reached target
    # --------------------------------------------------------

    if (
        len(class_counts) == 34
        and all(
            count >= SAMPLES_PER_CLASS
            for count in class_counts.values()
        )
    ):

        print(
            "\nAll 34 classes reached "
            f"{SAMPLES_PER_CLASS} samples."
        )

        break


# ============================================================
# SHOW COUNTS
# ============================================================

print("\n")
print("=" * 70)
print("COLLECTED SAMPLE COUNTS")
print("=" * 70)

for label in sorted(class_counts):

    print(
        f"{label:<35}"
        f"{class_counts[label]}"
    )


# ============================================================
# COMBINE TEMPORARY FILES
# ============================================================

print("\n")
print("=" * 70)
print("CREATING FINAL DATASET")
print("=" * 70)

final_parts = []

for label in sorted(class_counts):

    temp_file = os.path.join(
        TEMP_DIR,
        f"{label}.csv"
    )

    if not os.path.exists(temp_file):
        continue

    # Read only this class
    class_df = pd.read_csv(
        temp_file
    )

    # Safety limit
    class_df = class_df.head(
        SAMPLES_PER_CLASS
    )

    final_parts.append(
        class_df
    )

    print(
        f"{label:<35}"
        f"{len(class_df)} samples"
    )


# ============================================================
# FINAL COMBINATION
# ============================================================

print("\nCombining classes...")

df = pd.concat(
    final_parts,
    ignore_index=True
)

del final_parts


print(
    f"Before duplicate removal: "
    f"{df.shape}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    ignore_index=True
)

print(
    f"Duplicates removed: "
    f"{before - len(df)}"
)


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col != "Label"
]

for col in feature_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[feature_columns] = (
    df[feature_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

before = len(df)

df = df.dropna(
    ignore_index=True
)

print(
    f"Invalid rows removed: "
    f"{before - len(df)}"
)


# ============================================================
# FINAL DISTRIBUTION
# ============================================================

print("\nFinal class distribution:")

print(
    df["Label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "ciciot_5000_per_class.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\n")
print("=" * 70)

print(
    f"Final dataset shape: "
    f"{df.shape}"
)

print(
    f"Saved to: {output_path}"
)

print("=" * 70)


# ============================================================
# CLEAN TEMP FILES
# ============================================================

for filename in os.listdir(TEMP_DIR):

    filepath = os.path.join(
        TEMP_DIR,
        filename
    )

    try:
        os.remove(filepath)
    except:
        pass

try:
    os.rmdir(TEMP_DIR)
except:
    pass

print("\nTemporary files cleaned.")