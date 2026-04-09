from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# =========================
#  MongoDB Connection
# =========================

# Replace username, password, and cluster details
MONGO_URI = "mongodb+srv://irfan:Irfan9030@cluster0.y69o4gu.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)

DATABASE_NAME = "Irfan"
COLLECTION_NAME = "waferfault"


csv_path = "./notebooks/uci-secom.csv"

df = pd.read_csv(csv_path)
print(f"✅ CSV loaded. Shape: {df.shape}")


# =========================
#  Convert DataFrame → JSON
# =========================

# Convert dataframe to list of dictionaries
json_record = list(json.loads(df.T.to_json()).values())

print("✅ Data converted to JSON format")

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)