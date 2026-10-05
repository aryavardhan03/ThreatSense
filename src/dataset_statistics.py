import os
import glob
import pandas as pd


# ============================================================
# 1. CICIoT2023
# ============================================================

print("=" * 80)
print("CICIoT2023 STATISTICS")
print("=" * 80)

ciciot_path = "data/CICIoT2023/MERGED_CSV"

ciciot_files = glob.glob(
    os.path.join(ciciot_path, "*.csv")
)

ciciot_counts = {}

total_rows = 0

for i, file in enumerate(ciciot_files, start=1):

    print(
        f"Reading CICIoT file "
        f"{i}/{len(ciciot_files)}: "
        f"{os.path.basename(file)}"
    )

    for chunk in pd.read_csv(
        file,
        usecols=["Label"],
        chunksize=100_000
    ):

        total_rows += len(chunk)

        counts = chunk["Label"].value_counts()

        for label, count in counts.items():
            ciciot_counts[label] = (
                ciciot_counts.get(label, 0) + count
            )


print("\nTotal CICIoT rows:", f"{total_rows:,}")

print(
    "Number of classes:",
    len(ciciot_counts)
)

print("\nClass distribution:")

for label, count in sorted(
    ciciot_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label:<35} {count:,}")


# ============================================================
# 2. TON-IoT
# ============================================================

print("\n" + "=" * 80)
print("TON-IoT STATISTICS")
print("=" * 80)

ton_path = "data/TON_IoT/TON_Iot.csv"

ton_counts = {}
ton_type_counts = {}

ton_rows = 0

for chunk in pd.read_csv(
    ton_path,
    usecols=["label", "type"],
    chunksize=100_000
):

    ton_rows += len(chunk)

    for label, count in chunk["label"].value_counts().items():
        ton_counts[label] = (
            ton_counts.get(label, 0) + count
        )

    for attack_type, count in chunk["type"].value_counts().items():
        ton_type_counts[attack_type] = (
            ton_type_counts.get(attack_type, 0) + count
        )


print("Total TON-IoT rows:", f"{ton_rows:,}")

print("\n'label' distribution:")

for label, count in sorted(
    ton_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{str(label):<20} {count:,}")


print("\n' type ' distribution:")

for attack_type, count in sorted(
    ton_type_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{str(attack_type):<30} {count:,}")


# ============================================================
# 3. X-IIoTID
# ============================================================

print("\n" + "=" * 80)
print("X-IIoTID STATISTICS")
print("=" * 80)

xiiot_path = "data/X_IIoTID/X-IIoTID dataset.csv"

class1_counts = {}
class2_counts = {}
class3_counts = {}

xiiot_rows = 0

for chunk in pd.read_csv(
    xiiot_path,
    usecols=["class1", "class2", "class3"],
    chunksize=100_000
):

    xiiot_rows += len(chunk)

    for label, count in chunk["class1"].value_counts().items():
        class1_counts[label] = (
            class1_counts.get(label, 0) + count
        )

    for label, count in chunk["class2"].value_counts().items():
        class2_counts[label] = (
            class2_counts.get(label, 0) + count
        )

    for label, count in chunk["class3"].value_counts().items():
        class3_counts[label] = (
            class3_counts.get(label, 0) + count
        )


print("Total X-IIoTID rows:", f"{xiiot_rows:,}")


print("\nclass1 distribution:")

for label, count in sorted(
    class1_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{str(label):<35} {count:,}")


print("\nclass2 distribution:")

for label, count in sorted(
    class2_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{str(label):<35} {count:,}")


print("\nclass3 distribution:")

for label, count in sorted(
    class3_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{str(label):<35} {count:,}")


print("\n" + "=" * 80)
print("STATISTICS COMPLETE")
print("=" * 80)