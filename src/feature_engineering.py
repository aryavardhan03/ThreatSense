import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("Loading original feature data...")

X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
)

X_val = np.load(
    f"{DATA_DIR}/X_val.npy"
)

X_test = np.load(
    f"{DATA_DIR}/X_test.npy"
)


# ============================================================
# ORIGINAL FEATURE NAMES
# ============================================================

feature_names = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance"
]


# ============================================================
# VERIFY
# ============================================================

assert X_train.shape[1] == len(feature_names)
assert X_val.shape[1] == len(feature_names)
assert X_test.shape[1] == len(feature_names)

print("\nOriginal shapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ============================================================
# CONVERT NUMPY → DATAFRAME
# ============================================================

train_df = pd.DataFrame(
    X_train,
    columns=feature_names
)

val_df = pd.DataFrame(
    X_val,
    columns=feature_names
)

test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def create_engineered_features(df):

    result = df.copy()

    eps = 1e-6

    # --------------------------------------------------------
    # 1. TCP FLAG RELATIONSHIPS
    # --------------------------------------------------------

    result["syn_ack_ratio"] = (
        result["syn_count"]
        / (result["ack_count"] + eps)
    )

    result["ack_syn_ratio"] = (
        result["ack_count"]
        / (result["syn_count"] + eps)
    )

    result["fin_syn_ratio"] = (
        result["fin_count"]
        / (result["syn_count"] + eps)
    )

    result["rst_syn_ratio"] = (
        result["rst_count"]
        / (result["syn_count"] + eps)
    )

    # --------------------------------------------------------
    # 2. TCP FLAG TOTAL
    # --------------------------------------------------------

    result["tcp_flag_total"] = (
        result["fin_count"]
        + result["syn_count"]
        + result["rst_count"]
        + result["ack_count"]
    )

    # --------------------------------------------------------
    # 3. FLAG DENSITY
    # --------------------------------------------------------

    result["syn_density"] = (
        result["syn_count"]
        / (result["Number"] + eps)
    )

    result["ack_density"] = (
        result["ack_count"]
        / (result["Number"] + eps)
    )

    result["fin_density"] = (
        result["fin_count"]
        / (result["Number"] + eps)
    )

    result["rst_density"] = (
        result["rst_count"]
        / (result["Number"] + eps)
    )

    # --------------------------------------------------------
    # 4. PACKET SIZE RELATIONSHIPS
    # --------------------------------------------------------

    result["range_size"] = (
        result["Max"]
        - result["Min"]
    )

    result["max_avg_ratio"] = (
        result["Max"]
        / (result["AVG"] + eps)
    )

    result["avg_min_ratio"] = (
        result["AVG"]
        / (result["Min"] + eps)
    )

    result["std_avg_ratio"] = (
        result["Std"]
        / (result["AVG"] + eps)
    )

    # --------------------------------------------------------
    # 5. VARIANCE / DISPERSION
    # --------------------------------------------------------

    result["variance_avg_ratio"] = (
        result["Variance"]
        / (
            result["AVG"] ** 2
            + eps
        )
    )

    # --------------------------------------------------------
    # 6. TRAFFIC RATE RELATIONSHIP
    # --------------------------------------------------------

    result["rate_per_packet"] = (
        result["Rate"]
        / (result["Number"] + eps)
    )

    # --------------------------------------------------------
    # 7. INTER-ARRIVAL RELATIONSHIP
    # --------------------------------------------------------

    result["iat_per_packet"] = (
        result["IAT"]
        / (result["Number"] + eps)
    )

    # --------------------------------------------------------
    # 8. PROTOCOL ACTIVITY
    # --------------------------------------------------------

    result["transport_activity"] = (
        result["TCP"]
        + result["UDP"]
    )

    result["application_activity"] = (
        result["HTTP"]
        + result["HTTPS"]
        + result["DNS"]
        + result["Telnet"]
        + result["SMTP"]
        + result["SSH"]
        + result["IRC"]
    )

    return result


# ============================================================
# CREATE ENGINEERED FEATURES
# ============================================================

print("\nCreating engineered features...")

train_engineered = create_engineered_features(
    train_df
)

val_engineered = create_engineered_features(
    val_df
)

test_engineered = create_engineered_features(
    test_df
)


print("\nFeature counts:")
print(
    "Original features  :",
    len(feature_names)
)

print(
    "Engineered features:",
    train_engineered.shape[1]
)

print(
    "New features added :",
    train_engineered.shape[1]
    - len(feature_names)
)


# ============================================================
# CLEAN NUMERICAL VALUES
# ============================================================

print("\nChecking invalid values...")

for df in [
    train_engineered,
    val_engineered,
    test_engineered
]:

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


# ------------------------------------------------------------
# IMPORTANT:
# Fill missing values using TRAINING statistics only.
# ------------------------------------------------------------

train_medians = train_engineered.median()

train_engineered = train_engineered.fillna(
    train_medians
)

val_engineered = val_engineered.fillna(
    train_medians
)

test_engineered = test_engineered.fillna(
    train_medians
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train_engineered = (
    train_engineered.values.astype(
        np.float32
    )
)

X_val_engineered = (
    val_engineered.values.astype(
        np.float32
    )
)

X_test_engineered = (
    test_engineered.values.astype(
        np.float32
    )
)


# ============================================================
# STANDARDIZE
# ============================================================
#
# IMPORTANT:
# Fit scaler ONLY on training data.
# ============================================================

print("\nStandardizing engineered features...")

scaler = StandardScaler()

X_train_engineered = scaler.fit_transform(
    X_train_engineered
).astype(np.float32)

X_val_engineered = scaler.transform(
    X_val_engineered
).astype(np.float32)

X_test_engineered = scaler.transform(
    X_test_engineered
).astype(np.float32)


# ============================================================
# SAVE DATA
# ============================================================

train_path = (
    f"{DATA_DIR}/X_train_engineered.npy"
)

val_path = (
    f"{DATA_DIR}/X_val_engineered.npy"
)

test_path = (
    f"{DATA_DIR}/X_test_engineered.npy"
)

scaler_path = (
    f"{MODEL_DIR}/engineered_scaler.pkl"
)

feature_path = (
    f"{MODEL_DIR}/engineered_feature_names.pkl"
)


np.save(
    train_path,
    X_train_engineered
)

np.save(
    val_path,
    X_val_engineered
)

np.save(
    test_path,
    X_test_engineered
)

joblib.dump(
    scaler,
    scaler_path
)

joblib.dump(
    list(train_engineered.columns),
    feature_path
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print(
    "X_train engineered:",
    X_train_engineered.shape
)

print(
    "X_val engineered  :",
    X_val_engineered.shape
)

print(
    "X_test engineered :",
    X_test_engineered.shape
)

print("\nSaved:")
print(train_path)
print(val_path)
print(test_path)
print(scaler_path)
print(feature_path)

print("=" * 70)