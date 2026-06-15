import joblib
import pandas as pd

pipeline_rf = joblib.load("models/modelo_random_forest_2.pkl")
print("Pipeline steps:", pipeline_rf.steps)
preprocessing = pipeline_rf.named_steps["preprocessing"]
print("\nPreprocessing:", preprocessing)
if hasattr(preprocessing, "transformers_"):
    print("\nTransformers:")
    for name, trans, cols in preprocessing.transformers_:
        print(f"Name: {name}, Transformer: {trans}, Columns: {cols}")
