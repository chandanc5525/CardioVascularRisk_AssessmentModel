import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "cardiovascular.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "artifacts", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
