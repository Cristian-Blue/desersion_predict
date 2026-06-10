from services.shap_service import get_shape
from services.prediction_service import predict_student

def predictHandler(validated_data):
    data_student = validated_data['data_student']
    models = validated_data['models']
    
    result = predict_student(
        data_student.model_dump(),
        models
    )

    if int(result['prediction']) == 1:
        result['status'] = 'ACTIVO'
        result['detail'] = {}
    else:
        result['status'] = 'DESERTOR'
        result['detail'] = get_shape(validated_data)
    
    return result
