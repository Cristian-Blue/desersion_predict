import sys
import os

project_root = "/Users/bluemaster23/Documents/desarrollos/predictDesertion"
sys.path.append(project_root)
os.chdir(project_root)

from schemas.prediction_schema import validateContent
from handlers.predict_handler import predictHandler

student_data = {
    "COD_CARRERA": "10",
    "ANO": 2026,
    "SEMESTRE": 1,
    "SEXO": "F",
    "EDAD": 21,
    "PROMEDIO_ACUM": 4.2,
    "CRE_APROBADOS_TOTAL": 90.0,
    "CRE_REPROBADOSTOTAL": 2,
    "CRED_PERDIDOS": 4,
    "CRED_APROBADOS": 86,
    "CRED_MATRICULADOS": 18,
    "NIVEL": 5,
    "ALU_TESIS": "N",
    "ESTADO_CIVIL": "SOLTERO",
    "ESTRATO_SOCIAL": "3",
    "ESTRATO": "3",
    "NATU_COLEGIO": "OFICIAL",
    "DEPT_COLEGIO": "5",
    "MPIO_COLEGIO": "5001",
    "SANCION": 0,
    "PERIODO_SANCION": 0,
    "AM_REALIZADA": 0,
    "AN_DISCIPLINARIA": 0,
    "PSICOLOGIA": 0,
    "MEDICO": 1,
    "BECATRABAJO": 0,
    "REINTEGROS": 0,
    "CANCELA_MATERIA": 0,
    "DEUDA": 0,
    "DESCUENTO": 0,
    "CONDICION_DISCAPACIDAD": 0,
    "TIPO_DISCAPACIDAD": "0",
    "GRUPO_ETNICO": 0,
    "COMUNIDAD_NEGRA": 0,
    "NUMERO_HIJOS": 0,
    "REGISTRO_VICTIMA": "N",
    "TIPO_VICTIMA": "0",
    "ICFES": 320
}

for model_name in ["random_forest", "xgb"]:
    print(f"\n===== TESTING MODEL: {model_name} =====")
    
    # Single Student
    print("--- Test: Single Student ---")
    single_payload = {
        "model": model_name,
        "data": student_data
    }
    validated_single = validateContent(single_payload)
    result_single = predictHandler(validated_single)
    print("Single Prediction Status:", result_single["status"])
    print("Single Prediction Keys:", list(result_single.keys()))

    # Multiple Students (Array)
    print("\n--- Test: Multiple Students ---")
    multiple_payload = {
        "model": model_name,
        "data": [student_data, student_data]
    }
    validated_multiple = validateContent(multiple_payload)
    result_multiple = predictHandler(validated_multiple)
    print("Multiple Predictions Type:", type(result_multiple))
    print("Multiple Predictions Length:", len(result_multiple))
    for idx, res in enumerate(result_multiple):
        print(f"Student {idx+1} Prediction Status:", res["status"])
