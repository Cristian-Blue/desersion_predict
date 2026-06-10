import joblib
import shap

rf_model = joblib.load("models/modelo_random_forest_2.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")


pipeline_rf = joblib.load(
    "models/modelo_random_forest_2.pkl"
)
pipeline_xgb = joblib.load(
    "models/xgb_model.pkl"
)

classifier_rf = pipeline_rf.named_steps[
    "classifier"
]
classifier_xgb = pipeline_xgb.named_steps[
    "classifier"
]


explainer_rf = shap.TreeExplainer(
    classifier_rf
)

explainer_xgb = shap.TreeExplainer(
    classifier_xgb
)
