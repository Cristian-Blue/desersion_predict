# Predict Desertion (Predicción de Deserción Estudiantil)

API REST basada en Flask diseñada para predecir el riesgo de deserción de estudiantes universitarios. Utiliza modelos de Machine Learning (Random Forest y XGBoost) y explica las decisiones individuales de las predicciones utilizando valores SHAP (SHapley Additive exPlanations).

---

## 🚀 Características Principales

*   **Modelos Soportados**: Carga y ejecución de pipelines de **Random Forest** y **XGBoost** serializados con `joblib`.
*   **Explicabilidad Local**: Implementación de **SHAP** para indicar cuáles variables aumentan, reducen o favorecen la permanencia/deserción del estudiante.
*   **Validación de Datos**: Validación estricta del esquema de datos utilizando `pydantic` en el punto de entrada.
*   **Preparado para Producción**: Contenerizado con Docker y orquestado mediante Docker Compose, corriendo bajo un servidor WSGI productivo (`gunicorn`).

---

## 🛠️ Estructura del Proyecto

El proyecto sigue una estructura limpia y desacoplada en capas:

```text
├── app.py                     # Inicialización del servidor Flask y CORS
├── config.py                  # Variables de entorno y configuraciones generales
├── Dockerfile                 # Configuración de Docker para despliegue
├── docker-compose.yml         # Definición de servicios Docker Compose
├── requirements.txt           # Dependencias del sistema (scikit-learn, xgboost, shap, etc.)
├── handlers/
│   └── predict_handler.py     # Intermediario entre la ruta HTTP y los servicios
├── models/
│   ├── modelo_random_forest_2.pkl # Pipeline serializado de Random Forest
│   ├── xgb_model.pkl              # Pipeline serializado de XGBoost
│   └── features.json          # Listado oficial de características esperadas
├── routes/
│   └── prediction_routes.py   # Definición de blueprints y endpoints HTTP
├── schemas/
│   └── prediction_schema.py   # Modelos Pydantic y validaciones de payloads
├── services/
│   ├── prediction_service.py  # Servicio centralizador de predicciones
│   ├── prediction_factory.py  # Fábrica encargada de instanciar el modelo correcto
│   ├── shap_service.py        # Generación de explicaciones SHAP para la predicción
│   └── predictions/
│       ├── random_forest.py   # Adaptador del modelo Random Forest
│       └── xgb.py             # Adaptador del modelo XGBoost
└── utils/
    └── model_loader.py        # Cargador en memoria y generador de explicadores (shap.TreeExplainer)
```

---

## ⚙️ Requisitos Previos

*   Python 3.11+
*   Pip (administrador de paquetes de Python)
*   Docker y Docker Compose (opcional para despliegue local o productivo)

---

## 📦 Modelos Requeridos

Antes de iniciar la aplicación, asegúrate de colocar los archivos binarios de los modelos entrenados dentro de la carpeta `models/` (estos archivos no se suben al control de versiones por su tamaño):

*   **`models/modelo_random_forest_2.pkl`**: Pipeline y clasificadores del modelo Random Forest.
*   **`models/xgb_model.pkl`**: Pipeline y clasificadores del modelo XGBoost.

---

## 🔧 Instalación y Ejecución

### Opción 1: Desarrollo Local (Virtualenv)

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd predictDesertion
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usar: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Ejecutar el servidor Flask:**
   ```bash
   python app.py
   ```
   El servidor iniciará localmente en `http://127.0.0.1:5000`.

### Opción 2: Ejecución con Docker / Docker Compose

La aplicación incluye soporte listo para producción con Gunicorn.

1. **Construir y levantar la aplicación:**
   ```bash
   docker-compose up --build
   ```

2. **Detener los servicios:**
   ```bash
   docker-compose down
   ```
   El contenedor expone el puerto `5000` (`http://localhost:5000/api/v1/predict`).

---

## 📡 API Reference

### Realizar una Predicción

*   **Ruta:** `/api/v1/predict`
*   **Método:** `POST`
*   **Headers:** `Content-Type: application/json`

#### Payload de Entrada (Ejemplo)

El cuerpo de la petición debe contener el modelo a utilizar (`random_forest` o `xgb`) y un objeto `data` con la información del estudiante:

```json
{
  "model": "random_forest",
  "data": {
    "COD_CARRERA": "ING_SISTEMAS",
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
    "DEPT_COLEGIO": "ANTIOQUIA",
    "MPIO_COLEGIO": "MEDELLIN",
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
    "TIPO_DISCAPACIDAD": "NINGUNA",
    "GRUPO_ETNICO": 0,
    "COMUNIDAD_NEGRA": 0,
    "NUMERO_HIJOS": 0,
    "REGISTRO_VICTIMA": "N",
    "TIPO_VICTIMA": "NINGUNA",
    "ICFES": 320
  }
}
```

> [!NOTE]
> Todos los campos en `data` son validados a nivel de tipo de datos y rangos (por ejemplo, `EDAD` entre 15 y 100, `PROMEDIO_ACUM` entre 0.0 y 5.0) por el validador de `pydantic`.

#### Respuesta de Éxito (`200 OK`)

La respuesta devuelve el resultado de la predicción, la confianza asociada y los principales factores explicativos calculados mediante SHAP:

```json
{
  "prediction": 1,
  "confidence": 0.9254,
  "status": "ACTIVO",
  "reason": [
    {
      "feature": "PROMEDIO_ACUM",
      "value": 4.2,
      "importance": 0.234567,
      "direction": "FAVORECE_PERMANENCIA"
    },
    {
      "feature": "CRE_REPROBADOSTOTAL",
      "value": 2,
      "importance": 0.125432,
      "direction": "DISMINUYE_PERMANENCIA"
    },
    {
      "feature": "CRED_MATRICULADOS",
      "value": 18,
      "importance": 0.087654,
      "direction": "FAVORECE_PERMANENCIA"
    },
    {
      "feature": "EDAD",
      "value": 21,
      "importance": 0.045678,
      "direction": "FAVORECE_PERMANENCIA"
    },
    {
      "feature": "DEUDA",
      "value": 0,
      "importance": 0.034567,
      "direction": "FAVORECE_PERMANENCIA"
    }
  ]
}
```

*   **`prediction`**: Representa la clase predicha (`1` para Activo, `0` para Desertor).
*   **`status`**: Etiqueta legible del estado (`ACTIVO` o `DESERTOR`).
*   **`confidence`**: Nivel de certidumbre del modelo (entre 0.0 y 1.0).
*   **`reason`**: Lista de las variables con mayor influencia en la decisión final.
    *   `direction` puede ser `FAVORECE_PERMANENCIA` o `DISMINUYE_PERMANENCIA` si la predicción es 1 (`ACTIVO`), y `AUMENTA_RIESGO` o `REDUCE_RIESGO` si la predicción es 0 (`DESERTOR`).

#### Respuesta de Error (`400 Bad Request`)

Si el JSON recibido no cumple con el esquema definido por `StudentPredictionRequest`:

```json
{
  "error": [
    {
      "loc": ["data", "PROMEDIO_ACUM"],
      "msg": "Input should be less than or equal to 5",
      "type": "less_than_equal"
    }
  ]
}
```

---

## 📈 Explicabilidad y SHAP

El servicio de explicabilidad (`services/shap_service.py`) recupera los explicadores basados en árboles de decisión `shap.TreeExplainer` de cada modelo. 

1. El pipeline original realiza el preprocesamiento de variables categóricas y numéricas.
2. Se extraen los valores de impacto SHAP correspondientes a la clase predicha.
3. Se seleccionan los `top_n` factores cuya importancia (`abs(shap_value)`) sea superior a un umbral establecido (`0.0001`).
4. Se traduce el impacto cuantitativo en una dirección intuitiva de riesgo o permanencia.
