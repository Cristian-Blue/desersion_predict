import numpy as np
import pandas as pd


def to_python_type(value):
    """
    Convierte tipos de numpy/pandas a tipos nativos de Python
    para evitar errores de serialización JSON.
    """

    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def get_shap(data: dict, pipeline, explainer, top_n: int = 5):

    try:

        # DataFrame del estudiante
        df = pd.DataFrame(data)

        # Transformación igual al entrenamiento
        X_trans = pipeline.named_steps[
            "preprocessing"
        ].transform(df)

        # Nombres de variables transformadas
        feature_names = pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        # SHAP
        shap_values = explainer.shap_values(
            X_trans
        )

        values = np.array(shap_values)

        # Compatibilidad RandomForest / XGBoost
        if isinstance(shap_values, list):

            # Versiones antiguas de SHAP
            shap_class = np.array(
                shap_values[1]
            )[0]

        elif len(values.shape) == 3:

            # SHAP moderno multiclase
            shap_class = values[0, :, 1]

        else:

            # SHAP binario
            shap_class = values[0]

        result = []

        for feature, impact in zip(
            feature_names,
            shap_class
        ):

            feature_clean = feature

            # Elimina prefijos:
            # num__
            # cat__
            if "__" in feature_clean:
                feature_clean = feature_clean.split("__")[1]

            # Valor original
            value = None

            if feature_clean in df.columns:
                value = df.iloc[0][feature_clean]

            impact_value = round(
                float(impact),
                6
            )

            direction = (
                "AUMENTA_RIESGO"
                if impact_value < 0
                else "REDUCE_RIESGO"
            )

            result.append({
                "feature": feature_clean,
                "value": to_python_type(value),
                "impact": impact_value,
                "direction": direction
            })

        # Ordenar por importancia absoluta
        result.sort(
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        # Eliminar impactos insignificantes
        result = [
            item
            for item in result
            if abs(item["impact"]) > 0.0001
        ]

        return result[:top_n]

    except Exception as e:

        return [{
            "error": str(e)
        }]