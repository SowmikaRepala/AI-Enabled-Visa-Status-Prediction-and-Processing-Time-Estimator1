from flask import Flask, render_template, request
import pandas as pd
import joblib

# -------------------------------------------------
# App init
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Load trained model and lookup tables
# -------------------------------------------------
model = joblib.load("models/best_model.pkl")
nat_avg_map = joblib.load("models/nationality_avg_map.pkl")
visa_type_map = joblib.load("models/visa_type_map.pkl")

GLOBAL_AVG = sum(nat_avg_map.values()) / len(nat_avg_map)

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def get_season(month):
    if month in [12, 1]:
        return "Holiday"
    elif month in [6, 7, 8]:
        return "Peak"
    else:
        return "Normal"

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Raw user inputs
    user_input = {
        "visa_type": request.form["visa_type"],
        "destination_country": request.form["destination_country"],
        "nationality": request.form["nationality"],
        "processing_center": request.form["processing_center"],
        "interview": int(request.form["interview_required"]),
        "biometric": int(request.form["biometric_required"]),
        "mode": int(request.form["application_mode"]),
        "month": int(request.form["application_month"])
    }

    # -------- Feature engineering (server-side) --------
    features = {
        "visa_type_cleaned": visa_type_map.get(
            user_input["visa_type"], "Other"
        ),
        "destination_country": user_input["destination_country"],
        "nationality": user_input["nationality"],
        "processing_center": user_input["processing_center"],
        "interview_required_num": user_input["interview"],
        "biometric_required_num": user_input["biometric"],
        "application_mode_num": user_input["mode"],
        "application_month": user_input["month"],
        "season": get_season(user_input["month"]),
        "nationality_avg_processing": nat_avg_map.get(
            user_input["nationality"], GLOBAL_AVG
        )
    }

    input_df = pd.DataFrame([features])

    # Predict
    pred = int(model.predict(input_df)[0])

    return render_template(
        "result.html",
        prediction=f"{pred-2} to {pred+2} days"
    )


if __name__ == "__main__":
    app.run()