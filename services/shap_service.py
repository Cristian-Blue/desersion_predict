import numpy as np
import pandas as pd


def to_python_type(value):

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


def get_shap(
    data: dict,
    pipeline,
    explainer,
    prediction_class: int,
    top_n: int = 5,
    type_model: bool = True
):

    try:

        # DataFrame original
        if type_model:
            df = pd.DataFrame([data])
        else:
            df =  pd.DataFrame(data)

        # Transformación exactamente igual al entrenamiento
        X_trans = pipeline.named_steps[
            "preprocessing"
        ].transform(df)

        # Variables transformadas
        feature_names = pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        # SHAP
        shap_values = explainer.shap_values(
            X_trans
        )

        values = np.array(shap_values)

        print(
            "Prediction class:",
            prediction_class
        )

        print(
            "SHAP shape:",
            values.shape
        )

        # ------------------------
        # RANDOM FOREST ANTIGUO
        # ------------------------
        if isinstance(shap_values, list):

            shap_class = np.array(
                shap_values[prediction_class]
            )[0]

        # ------------------------
        # RANDOM FOREST NUEVO
        # shape=(1,n_features,2)
        # ------------------------
        elif len(values.shape) == 3:

            shap_class = values[
                0,
                :,
                prediction_class
            ]

        # ------------------------
        # XGBOOST BINARIO
        # shape=(1,n_features)
        # ------------------------
        elif len(values.shape) == 2:

            shap_class = values[0]

        # ------------------------
        # XGBOOST EXTRAÑO
        # shape=(n_features,)
        # ------------------------
        elif len(values.shape) == 1:

            shap_class = values

        else:

            raise Exception(
                f"Formato SHAP no soportado: {values.shape}"
            )

        result = []

        for feature, impact in zip(
            feature_names,
            shap_class
        ):

            feature_clean = feature

            if "__" in feature_clean:
                feature_clean = feature_clean.split("__")[1]

            value = None

            if feature_clean in df.columns:
                value = df.iloc[0][feature_clean]

            impact_value = round(
                float(impact),
                6
            )

            # Dirección según clase predicha
            if prediction_class == 0:

                direction = (
                    "AUMENTA_RIESGO"
                    if impact_value > 0
                    else "REDUCE_RIESGO"
                )

            else:

                direction = (
                    "FAVORECE_PERMANENCIA"
                    if impact_value > 0
                    else "DISMINUYE_PERMANENCIA"
                )

            result.append({
                "feature": feature_clean,
                "value": to_python_type(
                    value
                ),
                "importance": round(
                    abs(impact_value),
                    6
                ),
                "direction": direction
            })

        result.sort(
            key=lambda x: x[
                "importance"
            ],
            reverse=True
        )

        result = [
            item
            for item in result
            if item["importance"] > 0.0001
        ]

        return result[:top_n]

    except Exception as e:

        return [{
            "error": str(e)
        }]