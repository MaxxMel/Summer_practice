import os

# Директории проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Имена датасетов
DATASETS = ["Dataset1", "Dataset2", "dataset2_1", "dataset2_2", "Dataset5", "Dataset6"]
