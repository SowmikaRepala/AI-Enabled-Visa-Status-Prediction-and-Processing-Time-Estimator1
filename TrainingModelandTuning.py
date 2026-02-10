import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =================================================
# 1️⃣ LOAD DATASET
# =================================================
df = pd.read_csv("visa_dataset_ml_ready.csv")

print("Dataset loaded:", df.shape)


# =================================================
# 2️⃣ DEFINE FEATURES & TARGET
# =================================================
X = df.drop(columns=["processing_time_days"])
y = df["processing_time_days"]

categorical_features = [
    "visa_type_cleaned",
    "destination_country",
    "nationality",
    "processing_center",
    "season"
]

numeric_features = [
    "interview_required_num",
    "biometric_required_num",
    "application_mode_num",
    "application_month",
    "nationality_avg_processing"
]


# =================================================
# 3️⃣ PREPROCESSOR
# =================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# =================================================
# 4️⃣ TRAIN-TEST SPLIT
# =================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =================================================
# 5️⃣ DEFINE MODELS
# =================================================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    )
}


# =================================================
# 6️⃣ TRAIN & EVALUATE ALL MODELS
# =================================================
results = []
trained_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])
    trained_pipelines[name] = pipeline

    print(f"\n{name}")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2  : {r2:.3f}")


# =================================================
# 7️⃣ MODEL COMPARISON TABLE
# =================================================
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
).sort_values(by="RMSE")

print("\n📊 Model Comparison:")
print(results_df)


# =================================================
# 8️⃣ AUTO-SELECT BEST MODEL (LOWEST RMSE)
# =================================================
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

print(f"\n✅ Best baseline model: {best_model_name}")


# =================================================
# 9️⃣ TUNE ONLY IF BEST IS GRADIENT BOOSTING
# =================================================
if best_model_name == "Gradient Boosting":

    print("\n🔧 Tuning Gradient Boosting...")

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0]
    }

    grid = GridSearchCV(
        best_pipeline,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    tuned_model = grid.best_estimator_
    tuned_preds = tuned_model.predict(X_test)

    print("\n🎯 Tuned Gradient Boosting Results")
    print("MAE :", mean_absolute_error(y_test, tuned_preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, tuned_preds)))
    print("R2  :", r2_score(y_test, tuned_preds))

else:
    print("\nℹ️ Best model does not require tuning.")


print("\n✅ Training & selection completed.")

# =================================================
# COMPARISON: BASELINE vs TUNED MODELS
# =================================================
final_comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting (Baseline)",
        "Gradient Boosting (Tuned)"
    ],
    "MAE": [
        mean_absolute_error(y_test, trained_pipelines["Linear Regression"].predict(X_test)),
        mean_absolute_error(y_test, trained_pipelines["Random Forest"].predict(X_test)),
        mean_absolute_error(y_test, trained_pipelines["Gradient Boosting"].predict(X_test)),
        mean_absolute_error(y_test, tuned_preds)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, trained_pipelines["Linear Regression"].predict(X_test))),
        np.sqrt(mean_squared_error(y_test, trained_pipelines["Random Forest"].predict(X_test))),
        np.sqrt(mean_squared_error(y_test, trained_pipelines["Gradient Boosting"].predict(X_test))),
        np.sqrt(mean_squared_error(y_test, tuned_preds))
    ],
    "R2 Score": [
        r2_score(y_test, trained_pipelines["Linear Regression"].predict(X_test)),
        r2_score(y_test, trained_pipelines["Random Forest"].predict(X_test)),
        r2_score(y_test, trained_pipelines["Gradient Boosting"].predict(X_test)),
        r2_score(y_test, tuned_preds)
    ]
})

print("\n📊 Final Model Comparison (After Tuning):")
print(final_comparison_df.round(3))

os.makedirs("models", exist_ok=True)

joblib.dump(tuned_model, "models/best_model.pkl")

print("✅ Tuned Gradient Boosting model saved successfully")