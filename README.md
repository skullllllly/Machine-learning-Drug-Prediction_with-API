# Drug Recommendation System with FastAPI

A compact machine learning project that trains a drug-prescription prediction model from patient health metrics and exposes the model through a FastAPI endpoint.

## What it does

- Trains a decision tree regression pipeline on the `drug200.csv` dataset.
- Encodes categorical patient features and predicts the most likely prescribed drug.
- Serves inference requests through a REST-style FastAPI endpoint.
- Returns a clear JSON response with input values and the recommended drug.

## Key files

- `drug200.csv` — dataset containing patient features and prescribed drug labels.
- `ml_drug_withAPI.py` — training script that builds the preprocessing + model pipeline and exports two artifacts:
  - `drug_pipeline_model.pkl`
  - `label_encoder.pkl`
- `main.py` — FastAPI application that loads the saved artifacts and exposes a path-based inference route.

## How to use

1. Install dependencies

```bash
pip install fastapi uvicorn pandas scikit-learn joblib
```

2. Train the model (if artifacts are not already present)

```bash
python ml_drug_withAPI.py
```

3. Start the API server

```bash
uvicorn main:app --reload
```

4. Call the prediction endpoint

Open in browser or use `curl`:

```bash
http://127.0.0.1:8000/docs
```

## Example response

```json
{
  "status": "Success",
  "extracted_path_parameters": {
    "Age": 45,
    "Sex": "M",
    "BP": "HIGH",
    "Cholesterol": "NORMAL",
    "Na_to_K": 18.5
  },
  "recommended_drug": "DrugY"
}
```

> Replace `DrugY` with the actual model recommendation from your dataset.

## Notes

- The API uses path parameters for inference input.
- The model expects `Sex`, `BP`, and `Cholesterol` values in uppercase format.
- `Na_to_K` must be a positive float.

## Project goal

This repository demonstrates how to connect a trained Scikit-learn pipeline to a lightweight web API, making drug-prescription predictions available for real-time use.
