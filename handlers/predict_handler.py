from services.prediction_service import predict_student

def predictHandler(validated_data):
    data_students = validated_data['data_student']
    models = validated_data['models']
    is_list = validated_data['is_list']

    results = []
    for student in data_students:
        result = predict_student(
            student.model_dump(),
            models
        )
        results.append(result)

    return results if is_list else results[0]
