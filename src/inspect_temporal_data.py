import pandas as pd


DATA_PATH = "data/processed/ciciot_balanced.csv"

df = pd.read_csv(DATA_PATH)


print("Dataset shape:", df.shape)

print("\nAll columns:")
for i, column in enumerate(df.columns):
    print(i, "->", column)


print("\nPotential temporal/network-related columns:")

temporal_keywords = [
    "time",
    "timestamp",
    "date",
    "iat",
    "duration",
    "flow",
    "number"
]

for column in df.columns:
    if any(keyword in column.lower() for keyword in temporal_keywords):
        print("-", column)


print("\nFirst 10 rows of relevant columns:")

relevant_columns = [
    column
    for column in df.columns
    if any(keyword in column.lower() for keyword in temporal_keywords)
]

if relevant_columns:
    print(df[relevant_columns].head(10))
else:
    print("No obvious temporal columns found.")