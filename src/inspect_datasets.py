import os
import glob
import pandas as pd


# ============================================================
# Dataset paths
# ============================================================

DATASETS = {
    "CICIoT2023": "data/CICIoT2023/MERGED_CSV",
    "TON-IoT": "data/TON_IoT/TON_Iot.csv",
    "X-IIoTID": "data/X_IIoTID/X-IIoTID dataset.csv"
}


# ============================================================
# Inspect CICIoT2023
# ============================================================

print("=" * 80)
print("CICIoT2023")
print("=" * 80)

ciciot_files = glob.glob(
    os.path.join(
        DATASETS["CICIoT2023"],
        "*.csv"
    )
)

print("Number of files:", len(ciciot_files))

sample_file = ciciot_files[0]

df = pd.read_csv(
    sample_file,
    nrows=5
)

print("\nSample file:", sample_file)
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# Inspect TON-IoT
# ============================================================

print("\n" + "=" * 80)
print("TON-IoT")
print("=" * 80)

ton_path = DATASETS["TON-IoT"]

ton = pd.read_csv(
    ton_path,
    nrows=5
)

print("Number of columns:", len(ton.columns))

print("\nColumns:")
print(ton.columns.tolist())

print("\nFirst 5 rows:")
print(ton.head())


# ============================================================
# Inspect X-IIoTID
# ============================================================

print("\n" + "=" * 80)
print("X-IIoTID")
print("=" * 80)

xiiot_path = DATASETS["X-IIoTID"]

xiiot = pd.read_csv(
    xiiot_path,
    nrows=5
)

print("Number of columns:", len(xiiot.columns))

print("\nColumns:")
print(xiiot.columns.tolist())

print("\nFirst 5 rows:")
print(xiiot.head())