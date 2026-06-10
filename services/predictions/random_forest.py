import pandas as pd
from utils.model_loader import classifier_rf, explainer_rf
from services.shap_service import get_shap

def randomForest(data: pd.DataFrame):

    data = pd.DataFrame([data])

    prediction = classifier_rf.predict(data)[0]
    probability = classifier_rf.predict_proba(data)[0].max()
    result =  {
        "prediction": int(prediction),
        "confidence": round(float(probability),4)
    }

    if result["prediction"] == 1:
        result["status"] = "ACTIVO"
    else:
        result["status"] = "DESERTOR"
        result["reason"] = get_shap(data, classifier_rf, explainer_rf)

    return result