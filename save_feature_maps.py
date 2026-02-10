import pandas as pd
import joblib

df = pd.read_csv("visa_dataset_feature_engineered.csv")

# Nationality → avg processing
nat_avg = df.groupby("nationality")["processing_time_days"].mean().to_dict()

# Visa type cleaning
visa_map = {
    "Student": "Student",
    "Studying": "Student",
    "Study - student": "Student",
    "Work": "Work",
    "Working": "Work",
    "Tourist": "Tourist",
    "Visiting": "Visiting"
}

joblib.dump(nat_avg, "models/nationality_avg_map.pkl")
joblib.dump(visa_map, "models/visa_type_map.pkl")

print("✅ Feature lookup tables saved")
