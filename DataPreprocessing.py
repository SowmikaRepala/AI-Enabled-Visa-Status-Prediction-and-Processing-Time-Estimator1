import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load dataset (local file)
# -------------------------------
df = pd.read_csv("visa_applications_global_with_canada_realistic (1) (1).csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())

# -------------------------------
# Convert date columns
# -------------------------------
date_cols = ["application_date", "decision_date"]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -------------------------------
# Calculate processing_time_days if missing
# -------------------------------
mask = (
    df["processing_time_days"].isnull()
    & df["application_date"].notnull()
    & df["decision_date"].notnull()
)

df.loc[mask, "processing_time_days"] = (
    df.loc[mask, "decision_date"] - df.loc[mask, "application_date"]
).dt.days

print("Missing processing_time_days:", df["processing_time_days"].isnull().sum())

# -------------------------------
# Drop rows with missing target-related values
# -------------------------------
df = df.dropna(subset=["processing_time_days"])
print("After drop:", df["processing_time_days"].isnull().sum())

# -------------------------------
# Drop unnecessary column
# -------------------------------
if "processing_time_weeks" in df.columns:
    df = df.drop(columns=["processing_time_weeks"])

# -------------------------------
# Fill missing categorical values
# -------------------------------
df["application_mode"] = df["application_mode"].fillna("Unknown")

print(df.isnull().sum())

# -------------------------------
# Save cleaned dataset locally
# -------------------------------
df_cleaned = df.copy()
df_cleaned.to_csv("visa_dataset_cleaned.csv", index=False)
print("Cleaned dataset saved as visa_dataset_cleaned.csv")

# -------------------------------
# Identify categorical columns
# -------------------------------
categorical_cols = df.select_dtypes(include="object").columns
print("Categorical columns:", categorical_cols)

# -------------------------------
# Drop ID column
# -------------------------------
if "application_id" in df.columns:
    df = df.drop(columns=["application_id"])

# -------------------------------
# Encode binary columns
# -------------------------------
binary_cols = ["biometric_required", "interview_required"]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

# -------------------------------
# Encode target variable
# -------------------------------
print("Visa status values:", df["visa_status"].unique())

df["visa_status"] = df["visa_status"].map({"Approved": 1, "Rejected": 0})

# -------------------------------
# One-hot encoding for categorical variables
# -------------------------------
cols_to_encode = [
    "destination_country",
    "visa_type",
    "processing_center",
    "application_mode",
    "nationality"
]

existing_categorical_cols = [
    col for col in cols_to_encode
    if col in df.columns and df[col].dtype == "object"
]

if existing_categorical_cols:
    df = pd.get_dummies(
        df,
        columns=existing_categorical_cols,
        drop_first=True
    )
else:
    print("No categorical columns found for one-hot encoding.")

# -------------------------------
# Final checks
# -------------------------------
print(df.head())
print("Missing visa_status:", df["visa_status"].isna().sum())
print("Final missing values:\n", df.isnull().sum())

