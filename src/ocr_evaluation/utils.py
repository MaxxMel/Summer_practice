# ocr_evaluation/utils.py
import cv2
import logging
import numpy as np
import ast
import os

def resize_image(image, max_size=1024):
    """Изменение размера изображения с сохранением пропорций."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def parse_bbox(bbox_str, img_width=None, img_height=None, dataset_name=None):
    """Парсинг bbox в зависимости от формата датасета."""
    try:

        if dataset_name in ['Text_detection_in_the_documents', 'Total_text']:
            return None
        elif dataset_name == 'sbernotes':
            x, y, w, h = map(float, bbox_str.strip('"').split(','))
            if w <= 0 or h <= 0:
                logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
                return None
            return (int(x), int(y), int(w), int(h))
        elif dataset_name in ['russian_language_visual_text_recognition', 'licance_plate_characters_detection_ocr', 'ocr_machine_readable_zone_mrz_detection']:
            bbox = ast.literal_eval(bbox_str)
            if dataset_name == 'russian_language_visual_text_recognition':
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
                    return None
                x = int(x * img_width)
                y = int(y * img_height)
                w = int(w * img_width)
                h = int(h * img_height)
                return (x, y, w, h)
            else:  # Dataset5, Dataset6
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                if w <= 0 or h <= 0:
                    logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
                    return None
                return (x, y, w, h)
    except Exception as e:
        logging.error(f'Ошибка парсинга bbox: {bbox_str}, {e}')
        return None

def clean_ground_truth(text):
    """Очистка текста ground truth."""
    import pandas as pd
    if pd.isna(text):
        return ''
    text = str(text).strip()
    if text.endswith(','):
        text = text[:-1]
    return text
