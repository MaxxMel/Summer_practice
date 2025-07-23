import os

# Директории проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Имена датасетов
DATASETS = ["NT", "d1", "d2", "d2_1", "d2_2", "d5", "d6"]
