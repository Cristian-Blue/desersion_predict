from services.prediction_service import predict_student

def predictHandler(validated_data):
    data_students = validated_data['data_student']
    models = validated_data['models']
    is_list = validated_data['is_list']

    results = []
    for student in data_students:
        student_copy = student.model_dump()
        student_copy.pop('CODIGO_ALUMNO')
        result = predict_student(
            student_copy,
            models
        )
        result['CODIGO_ALUMNO'] = student.CODIGO_ALUMNO
        results.append(result)

    return results if is_list else results[0]
