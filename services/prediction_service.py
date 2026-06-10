from services.prediction_factory import predictionFactory 

def predict_student(data, models ):

    result = predictionFactory(models, data)

    return result