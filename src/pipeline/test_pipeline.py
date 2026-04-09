import os, sys
import pandas as pd
import numpy as np
import shap

from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


# -----------------------------
# 📁 Output File Details
# -----------------------------
@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "predicted_file.csv"
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)


# -----------------------------
# 🚀 Prediction Pipeline
# -----------------------------
class PredictionPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.utils = MainUtils()
        self.prediction_file_detail = PredictionFileDetail()

    # -----------------------------
    # 🤖 Prediction + SHAP Explainability
    # -----------------------------
    def predict(self, features: pd.DataFrame):
        try:
            model = self.utils.load_object(os.path.join(artifact_folder, "model.pkl"))
            preprocessor = self.utils.load_object(os.path.join(artifact_folder, "preprocessor.pkl"))

            transformed_x = preprocessor.transform(features)
            preds = model.predict(transformed_x)

            explanations = []

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(transformed_x)

                if isinstance(shap_values, list):
                    vals = shap_values[1]
                else:
                    vals = shap_values

                for i, prediction in enumerate(preds):
                    if prediction == 0:
                        row_values = vals[i]
                        top_feature_idx = np.argmax(row_values)
                        top_feature_impact = row_values[top_feature_idx]

                        feature_name = features.columns[top_feature_idx]
                        reason = f"{feature_name} (Impact: {top_feature_impact:.2f})"

                        explanations.append(reason)
                    else:
                        explanations.append("N/A (Good Wafer)")

            except Exception as e:
                logging.info(f"SHAP Error: {str(e)}")
                explanations = ["Explanation Unavailable"] * len(preds)

            return preds, explanations

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # 📊 Create Output File
    # -----------------------------
    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            input_df = pd.read_csv(input_dataframe_path)

            predictions, explanations = self.predict(input_df)

            input_df[TARGET_COLUMN] = predictions

            target_mapping = {0: 'bad', 1: 'good'}
            input_df[TARGET_COLUMN] = input_df[TARGET_COLUMN].map(target_mapping)

            input_df["Root_Cause_Analysis"] = explanations

            os.makedirs(self.prediction_file_detail.prediction_output_dirname, exist_ok=True)
            input_df.to_csv(self.prediction_file_detail.prediction_file_path, index=False)

            logging.info("Prediction completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # ▶️ Run Full Pipeline
    # -----------------------------
    def run_pipeline(self):
        try:
            self.get_predicted_dataframe(self.file_path)
            return self.prediction_file_detail
        except Exception as e:
            raise CustomException(e, sys)