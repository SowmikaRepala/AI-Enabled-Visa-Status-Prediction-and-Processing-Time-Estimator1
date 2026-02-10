import joblib

nat_avg_map = joblib.load("models/nationality_avg_map.pkl")
visa_type_map = joblib.load("models/visa_type_map.pkl")

def get_season(month):
    if month in [12, 1]:
        return "Holiday"
    elif month in [6, 7, 8]:
        return "Peak"
    else:
        return "Normal"

def build_features(user_input):
    return {
        "visa_type_cleaned": visa_type_map.get(user_input["visa_type"], "Other"),
        "destination_country": user_input["destination_country"],
        "nationality": user_input["nationality"],
        "processing_center": user_input["processing_center"],
        "interview_required_num": user_input["interview_required"],
        "biometric_required_num": user_input["biometric_required"],
        "application_mode_num": user_input["application_mode"],
        "application_month": user_input["application_month"],
        "season": get_season(user_input["application_month"]),
        "nationality_avg_processing": nat_avg_map.get(
            user_input["nationality"], 
            sum(nat_avg_map.values()) / len(nat_avg_map)
        )
    }
