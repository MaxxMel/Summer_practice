# ocr_evaluation/config.py
import os

DATA_DIR = '/Users/maksim/Desktop/TETE/Summer_practice/data'
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATASETS = {
    'Text_detection_in_the_documents': {'csv': 'text_detection_in_the_documents_test.csv', 'bbox': False, 'lang': ['en', 'ru'], 'allowlist': None, 'image_path_column': 'image'},
    'Total_text': {'csv': 'total_text_test.csv', 'bbox': False, 'lang': ['en', 'ru'], 'allowlist': None, 'image_path_column': 'Image_path'},
    'sbernotes': {'csv': 'sbernotes_test.csv', 'bbox': True, 'lang': ['ru'], 'allowlist': 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя ', 'image_path_column': 'img_path'},
    'russian_language_visual_text_recognition': {'csv': 'russian_language_visual_text_recognition_test.csv', 'bbox': True, 'lang': ['en'], 'allowlist': None, 'image_path_column': 'img_path'},
    'licance_plate_characters_detection_ocr': {'csv': 'licance_plate_characters_detection_ocr_test.csv', 'bbox': True, 'lang': ['en'], 'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'image_path_column': 'Image_path'},
    'ocr_machine_readable_zone_mrz_detection': {'csv': 'ocr_machine_readable_zone_mrz_detection_test.csv', 'bbox': True, 'lang': ['en'], 'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<', 'image_path_column': 'Image_path'}
}
