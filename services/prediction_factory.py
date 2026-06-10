
from services.predictions.random_forest import randomForest
from services.predictions.xgb import xgb

def predictionFactory(model_name, data):
    if model_name == "random_forest":
        return randomForest(data)
    elif model_name == "xgb":
        return xgb(data)