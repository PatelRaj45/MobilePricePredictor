import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("best_mobile_price_model.pkl")  # Must use VotingClassifier with voting='soft'
scaler = joblib.load("scaler.pkl")

def predict_price_range(sample_df):
    # Add derived features
    sample_df["pixel_density"] = sample_df["px_height"] * sample_df["px_width"]
    sample_df["screen_area"] = sample_df["sc_h"] * sample_df["sc_w"]
    sample_df["camera_quality"] = sample_df["pc"] + sample_df["fc"]

    # Scale features
    sample_scaled = scaler.transform(sample_df)

    # Predict class
    pred_class = model.predict(sample_scaled)[0]

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        pred_probs = model.predict_proba(sample_scaled)[0]
    else:
        pred_probs = [1.0 if i == pred_class else 0.0 for i in range(4)]

    return pred_class, pred_probs
