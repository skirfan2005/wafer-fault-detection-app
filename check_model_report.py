import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.utils.main_utils import MainUtils
from src.constant import artifact_folder, TARGET_COLUMN

utils = MainUtils()

# ================= PATHS =================

MODEL_PATH = os.path.join(artifact_folder, "model.pkl")
PREPROCESSOR_PATH = os.path.join(artifact_folder, "preprocessor.pkl")

# 👉 CHANGE THIS TO YOUR TRAINING CSV
DATA_PATH = r"C:\Users\shaik\OneDrive\Desktop\FinalYearProject\artifacts\wafer_fault.csv"


# =======================================

print("\nLoading model...")
model = utils.load_object(MODEL_PATH)

print("Model Used:", type(model).__name__)

print("\nLoading preprocessor...")
preprocessor = utils.load_object(PREPROCESSOR_PATH)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

# same preprocessing as training
df.rename(columns={"Pass/Fail": TARGET_COLUMN}, inplace=True)
df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({-1: 0, 1: 1})

X = df.drop(columns=[TARGET_COLUMN, "Wafer"], errors="ignore")
y = df[TARGET_COLUMN]

# Drop ID / non numeric
X = X.select_dtypes(exclude=["object"])


print("\nTransforming features...")
X_transformed = preprocessor.transform(X)

print("Predicting...")
preds = model.predict(X_transformed)

# ================= METRICS =================

accuracy = accuracy_score(y, preds)
precision = precision_score(y, preds)
recall = recall_score(y, preds)
f1 = f1_score(y, preds)

print("\n================ MODEL PERFORMANCE ================\n")

print("Model Used :", type(model).__name__)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nDetailed Classification Report:\n")
print(classification_report(y, preds))

print("\n==================================================\n")
