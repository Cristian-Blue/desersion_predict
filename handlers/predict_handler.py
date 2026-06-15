from services.prediction_service import predict_student

def predictHandler(validated_data):
    data_student = validated_data['data_student']
    models = validated_data['models']
    result = predict_student(
        data_student.model_dump(),
        models
    )

    return result
