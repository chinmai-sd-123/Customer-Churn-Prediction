from fastapi import FastAPI,HTTPException
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from app.schemas import CustomerInput

app = FastAPI(title="Bank Customer Churn Prediction ")

model = load_model("Models/churn_ann_model.h5", compile=False)
scaler = joblib.load("Models/scaler.pkl")

FEATURE_ORDER= [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
    'Geography_Germany',
    'Geography_Spain',
    'Gender_Male'
]

threshold=0.50

@app.get("/")

def health_check():
    return {"status": "api is running"}

@app.post("/predict")

def predict_churn(data: CustomerInput):

    X_df=pd.DataFrame([data.model_dump()], columns=FEATURE_ORDER)
    X_df_scaled=scaler.transform(X_df)
    prob= model.predict(X_df_scaled)[0][0]

    if np.isnan(prob) or np.isinf(prob):
        raise HTTPException(status_code=400, detail="Model produced an invalid prediction. Check input values")

    return {"churn_probability": float(prob),
            "churn": int(prob >= threshold),
            }


