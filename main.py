import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Path

# 1. Initialize application with metadata
app = FastAPI(
    title="Drug Recommendation System API",
    description="URL Path-driven interface to predict prescribed drugs based on patient metrics.",
    version="2.0.0"
)

# 2. Load the pre-trained artifacts securely
try:
    pipeline = joblib.load('drug_pipeline_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    raise RuntimeError("Critical Error: Model artifacts missing from directory.") from e

# 3. Define the inference route using descriptive path parameters
@app.get(
    "/recommend-drug/{age}/{sex}/{bp}/{cholesterol}/{na_to_k}", 
    tags=["Path-Based Inference"]
)
async def recommend_drug_via_path(
    age: int = Path(..., description="Age of the patient", ge=0, le=120, example=45),
    sex: str = Path(..., description="Gender of the patient: 'F' or 'M'", example="M"),
    bp: str = Path(..., description="Blood Pressure level: 'HIGH', 'NORMAL', or 'LOW'", example="HIGH"),
    cholesterol: str = Path(..., description="Cholesterol level: 'HIGH' or 'NORMAL'", example="NORMAL"),
    na_to_k: float = Path(..., description="Sodium-to-Potassium ratio in blood", gt=0.0, example=18.5)
):
    try:
        # Standardize inputs to match capitalization during model training
        input_data = pd.DataFrame([{
            "Age": age,
            "Sex": sex.upper().strip(),
            "BP": bp.upper().strip(),
            "Cholesterol": cholesterol.upper().strip(),
            "Na_to_K": na_to_k
        }])
        
        # Execute Scikit-Learn Pipeline transformations and prediction
        encoded_prediction = pipeline.predict(input_data)
        
        # Decode numeric index back to the exact drug string name
        target_index = int(round(encoded_prediction[0]))
        predicted_drug = label_encoder.inverse_transform([target_index])[0]
        
        return {
            "status": "Success",
            "extracted_path_parameters": {
                "Age": age,
                "Sex": input_data["Sex"].iloc[0],
                "BP": input_data["BP"].iloc[0],
                "Cholesterol": input_data["Cholesterol"].iloc[0],
                "Na_to_K": na_to_k
            },
            "recommended_drug": str(predicted_drug)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Inference execution failure: {str(e)}"
        )