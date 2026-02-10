import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Load cleaned dataset (LOCAL)
# -------------------------------------------------
df = pd.read_csv("visa_dataset_cleaned.csv")

# -------------------------------------------------
# Basic inspection
# -------------------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -------------------------------------------------
# Distribution of processing time
# -------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["processing_time_days"], bins=30, kde=True)
plt.title("Distribution of Processing Time (Days)")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df["processing_time_days"])
plt.title("Processing Time Box Plot")
plt.show()

# -------------------------------------------------
# Visa type distribution
# -------------------------------------------------
df["visa_type"].value_counts().plot(kind="bar")
plt.title("Visa Type Distribution")
plt.show()

# -------------------------------------------------
# Clean visa type categories
# -------------------------------------------------
visa_map = {
    "Student": "Student",
    "Studying": "Student",
    "Study - student": "Student",
    "Work": "Work",
    "Working": "Work",
    "Tourist": "Tourist",
    "Visiting": "Visiting"
}

df["visa_type_cleaned"] = df["visa_type"].map(visa_map)

print("\nCleaned Visa Type Counts:")
print(df["visa_type_cleaned"].value_counts())

df["visa_type_cleaned"].value_counts().plot(kind="bar")
plt.title("Cleaned Visa Type Distribution")
plt.show()

# -------------------------------------------------
# Processing time vs visa type
# -------------------------------------------------
plt.figure(figsize=(7, 4))
sns.boxplot(
    x="visa_type_cleaned",
    y="processing_time_days",
    data=df
)
plt.title("Processing Time vs Visa Type")
plt.xticks(rotation=45)
plt.show()

# -------------------------------------------------
# Country vs processing time
# -------------------------------------------------
plt.figure(figsize=(8, 4))
sns.boxplot(
    x="destination_country",
    y="processing_time_days",
    data=df
)
plt.title("Country vs Processing Time")
plt.xticks(rotation=45)
plt.show()

# -------------------------------------------------
# Visa status distribution (count + percentage)
# -------------------------------------------------
status_counts = df["visa_status"].value_counts()
status_percent = df["visa_status"].value_counts(normalize=True) * 100

plt.figure(figsize=(5, 4))
bars = plt.bar(status_counts.index, status_counts.values)

for bar, pct in zip(bars, status_percent):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{pct:.1f}%",
        ha="center",
        va="bottom"
    )

plt.title("Visa Status Distribution")
plt.ylabel("Count")
plt.show()

# -------------------------------------------------
# Status vs processing time
# -------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(
    x="visa_status",
    y="processing_time_days",
    data=df
)
plt.title("Visa Status vs Processing Time")
plt.show()

# -------------------------------------------------
# Encode numerical features
# -------------------------------------------------
df["interview_required_num"] = df["interview_required"].map({"Yes": 1, "No": 0})
df["biometric_required_num"] = df["biometric_required"].map({"Yes": 1, "No": 0})
df["application_mode_num"] = df["application_mode"].map({
    "Online": 1,
    "Offline": 0,
    "Unknown": -1
})

df["application_date"] = pd.to_datetime(df["application_date"])
df["application_month"] = df["application_date"].dt.month

# -------------------------------------------------
# Correlation heatmap
# -------------------------------------------------
plt.figure(figsize=(8, 6))

corr = df[
    [
        "processing_time_days",
        "interview_required_num",
        "biometric_required_num",
        "application_mode_num",
        "application_month"
    ]
].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap: Engineered Numerical Features")
plt.show()

# -------------------------------------------------
# Feature engineering
# -------------------------------------------------
df["dest_country_avg_processing"] = (
    df.groupby("destination_country")["processing_time_days"]
      .transform("mean")
)

df["visa_type_avg_processing"] = (
    df.groupby("visa_type")["processing_time_days"]
      .transform("mean")
)

df["season"] = df["application_month"].map({
    12: "Holiday", 1: "Holiday",
    6: "Peak", 7: "Peak", 8: "Peak",
    2: "Normal", 3: "Normal", 4: "Normal", 5: "Normal",
    9: "Normal", 10: "Normal", 11: "Normal"
})

df["nationality_avg_processing"] = (
    df.groupby("nationality")["processing_time_days"]
      .transform("mean")
)

# -------------------------------------------------
# Season vs processing time
# -------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(
    x="season",
    y="processing_time_days",
    data=df
)
plt.title("Season vs Processing Time")
plt.show()

# -------------------------------------------------
# Save feature-engineered dataset
# -------------------------------------------------

# Drop columns NOT required for ML
columns_to_drop = [
    "application_id",
    "decision_date",
    "application_date",
    "visa_type",
    "visa_status",
    "dest_country_avg_processing",
    "visa_type_avg_processing"
]

df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

# -------------------------------------------------
# Select final ML features
# -------------------------------------------------
ml_features = [
    "visa_type_cleaned",
    "destination_country",
    "nationality",
    "processing_center",
    "interview_required_num",
    "biometric_required_num",
    "application_mode_num",
    "application_month",
    "season",
    "nationality_avg_processing",
    "processing_time_days"
]

df_ml = df[ml_features]

# -------------------------------------------------
# Final sanity check
# -------------------------------------------------
print("\nFinal ML-ready dataset shape:", df_ml.shape)
print(df_ml.head())

# -------------------------------------------------
# Save ML-ready dataset
# -------------------------------------------------
df_ml.to_csv(
    "visa_dataset_ml_ready.csv",
    index=False
)

print("\n✅ ML-ready dataset saved as visa_dataset_ml_ready.csv")