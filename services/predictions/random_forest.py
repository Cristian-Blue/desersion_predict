from utils.model_loader import pipeline_rf
import pandas as pd
from utils.model_loader import explainer_rf
from services.shap_service import get_shap

def randomForest(data: pd.DataFrame):

    data = pd.DataFrame([data])
    prediction = pipeline_rf.predict(data)[0] 
    probability = pipeline_rf.predict_proba(data)[0].max()

    result =  {
        "prediction": int(prediction),
        "confidence": round(float(probability),4)
    }

    if result["prediction"] == 1:
        result["status"] = "ACTIVO"
    else:
        result["status"] = "DESERTOR"
    result["reason"] = get_shap(
        data=data,
        pipeline=pipeline_rf,
        explainer=explainer_rf,
        prediction_class=int(prediction),
        top_n=5,
        type_model=False
    )
    return result