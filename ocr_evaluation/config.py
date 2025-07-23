import os

# Директории проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Имена датасетов
DATASETS = ["d1", "d2_1", "d3", "d4", "d5", "d6"]

# Параметры обработки
CHUNK_SIZE = 50
SAVE_INTERVAL = 10
MAX_IMAGE_SIZE = 512
MAX_CROP_SIZE = 128
BATCH_SIZE = 2

# Разрешенные символы для EasyOCR
CYRILLIC_ALLOWLIST = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя "
ALPHA_NUMERIC_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
