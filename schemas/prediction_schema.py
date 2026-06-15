# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Literal

class StudentPredictionRequest(BaseModel):

    COD_CARRERA: str

    ANO: int = Field(..., ge=2010, le=2100)

    SEMESTRE: int = Field(..., ge=1, le=2)

    SEXO: Literal["M", "F"]

    EDAD: int = Field(..., ge=15, le=100)

    PROMEDIO_ACUM: float = Field(..., ge=0, le=5)

    CRE_REPROBADOSTOTAL: int = Field(..., ge=0)

    CRED_PERDIDOS: int = Field(..., ge=0)

    CRED_APROBADOS: int = Field(..., ge=0)

    CRED_MATRICULADOS: int = Field(..., ge=0)

    NIVEL: int = Field(..., ge=1)

    ALU_TESIS: Literal["S", "N"]

    ESTADO_CIVIL: str

    ESTRATO_SOCIAL: str

    ESTRATO: str

    NATU_COLEGIO: str

    DEPT_COLEGIO: str

    MPIO_COLEGIO: str

    SANCION: int = Field(..., ge=0)

    PERIODO_SANCION: int = Field(..., ge=0)

    AM_REALIZADA: int = Field(..., ge=0)

    AN_DISCIPLINARIA: int = Field(..., ge=0)

    PSICOLOGIA: int = Field(..., ge=0)

    MEDICO: int = Field(..., ge=0)

    BECATRABAJO: int = Field(..., ge=0)

    REINTEGROS: int = Field(..., ge=0)

    CANCELA_MATERIA: int = Field(..., ge=0)

    DEUDA: int = Field(..., ge=0)

    DESCUENTO: int = Field(..., ge=0)

    CONDICION_DISCAPACIDAD: int = Field(..., ge=0)

    TIPO_DISCAPACIDAD: str

    GRUPO_ETNICO: int = Field(..., ge=0)

    COMUNIDAD_NEGRA: int = Field(..., ge=0)

    NUMERO_HIJOS: int = Field(..., ge=0)

    REGISTRO_VICTIMA: Literal["S", "N"]

    TIPO_VICTIMA: str

    ICFES: int = Field(..., ge=0)
    CRE_APROBADOS_TOTAL : float = Field(..., ge=0)

def validateContent(body):
    data = body['data']
    models = body['model']
    expected_features =["COD_CARRERA", "SEMESTRE", "SEXO", "EDAD", "PROMEDIO_ACUM", "CRE_APROBADOS_TOTAL", "CRE_REPROBADOSTOTAL", "CRED_PERDIDOS", "CRED_APROBADOS", "CRED_MATRICULADOS", "NIVEL", "ALU_TESIS", "ESTADO_CIVIL", "ESTRATO_SOCIAL", "ESTRATO", "NATU_COLEGIO", "DEPT_COLEGIO", "MPIO_COLEGIO", "SANCION", "PERIODO_SANCION", "AM_REALIZADA", "AN_DISCIPLINARIA", "PSICOLOGIA", "MEDICO", "BECATRABAJO", "REINTEGROS", "CANCELA_MATERIA", "DEUDA", "DESCUENTO", "CONDICION_DISCAPACIDAD", "TIPO_DISCAPACIDAD", "GRUPO_ETNICO", "COMUNIDAD_NEGRA", "NUMERO_HIJOS", "REGISTRO_VICTIMA", "TIPO_VICTIMA", "ICFES"]

    received = set(data.keys())
    expected_set = set(expected_features)

    missing = expected_set - received
    extra = received - expected_set
    
    return {
        "valid" : len(missing) == 0 and len(extra) == 0,
        "missing": list(missing),
        "extra": list(extra),
        "all_features": expected_features,
        "data_student": StudentPredictionRequest(**data),
        "models": models
    }
    