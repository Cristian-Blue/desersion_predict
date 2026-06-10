import numpy as np
import pandas as pd


def get_shap(pipeline, explainer, data: dict, top_n: int = 5):

    try:

        # Crear DataFrame
        df = pd.DataFrame([data])

        # Transformar igual que entrenamiento
        X_trans = pipeline.named_steps[
            "preprocessing"
        ].transform(df)

        # Obtener nombres de variables
        feature_names = pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        # Calcular SHAP
        shap_values = explainer.shap_values(
            X_trans
        )

        # Compatibilidad RF / XGB
        if isinstance(shap_values, list):

            shap_class = shap_values[1][0]

        elif len(np.array(shap_values).shape) == 3:

            shap_class = shap_values[:, :, 1][0]

        else:

            shap_class = shap_values[0]

        result = []

        for feature, impact in zip(
            feature_names,
            shap_class
        ):

            feature_clean = feature

            if "__" in feature:
                feature_clean = feature.split("__")[1]

            result.append({
                "feature": feature_clean,
                "impact": round(
                    float(impact),
                    6
                )
            })

        # Ordenar por importancia absoluta
        result = sorted(
            result,
            key=lambda x: abs(
                x["impact"]
            ),
            reverse=True
        )

        return result[:top_n]

    except Exception as e:

        return [{
            "error": str(e)
        }]