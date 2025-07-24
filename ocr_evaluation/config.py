"""import os

# Директории проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Имена датасетов
DATASETS = ["Dataset1", "Dataset2", "dataset2_1", "dataset2_2", "Dataset5", "Dataset6"]
"""
# ocr_evaluation/config.py
import os

DATA_DIR = '/Users/maksim/Desktop/TETE/Summer_practice/data'
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATASETS = {
    'Dataset1': {'csv': 'dataset1_test.csv', 'bbox': False, 'lang': ['en', 'ru'], 'allowlist': None, 'image_path_column': 'image'},
    'Dataset2': {'csv': 'dataset2_test.csv', 'bbox': False, 'lang': ['en', 'ru'], 'allowlist': None, 'image_path_column': 'Image_path'},
    'dataset2_1': {'csv': 'dataset2_1_test.csv', 'bbox': True, 'lang': ['ru'], 'allowlist': 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя ', 'image_path_column': 'img_path'},
    'dataset2_2': {'csv': 'dataset2_2_test.csv', 'bbox': True, 'lang': ['en'], 'allowlist': None, 'image_path_column': 'img_path'},
    'Dataset5': {'csv': 'dataset5_test.csv', 'bbox': True, 'lang': ['en'], 'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'image_path_column': 'Image_path'},
    'Dataset6': {'csv': 'dataset6_test.csv', 'bbox': True, 'lang': ['en'], 'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<', 'image_path_column': 'Image_path'}
}
