from utils.model_loader import explainer_xgb
from utils.model_loader import classifier_xgb
from services.shap_service import get_shap
import pandas as pd
from utils.model_loader import xgb_model

def xgb(data: dict):
    df = pd.DataFrame([data])
    
    prediction = xgb_model.predict(df)[0]
    probability = xgb_model.predict_proba(df)[0].max()
    result = {
        "prediction": int(prediction),
        "confidence": round(float(probability),4)
    }
    if result["prediction"] == 1:
        result["status"] = "ACTIVO"
    else:
        result["status"] = "DESERTOR"
        result["reason"] = get_shap(data, classifier_xgb, explainer_xgb)
    return result