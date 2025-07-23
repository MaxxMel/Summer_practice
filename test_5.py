
#!/usr/bin/env python3


import pandas as pd
import easyocr
import cv2
import jiwer
import logging
import numpy as np
import os
from tqdm import tqdm
import ast
from ocr_evaluation.config import DATA_DIR, PROCESSED_DATA_DIR

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(image, max_size=1024):

    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def parse_bbox(bbox_str):
    try:
        bbox = ast.literal_eval(bbox_str)
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

def evaluate_easyocr(csv_path):
    try:
        reader = easyocr.Reader(['en'], gpu=True)  # Только английский для буквенно-цифровых символов
        logging.info('EasyOCR инициализирован с поддержкой GPU')
    except Exception as e:
        logging.warning(f'GPU не поддерживается: {e}. Используется CPU.')
        reader = easyocr.Reader(['en'], gpu=False)

    # Чтение датасета
    df = pd.read_csv(csv_path)
    logging.info(f'Загружен датасет с {len(df)} записями')

    wers = []
    cers = []

    # Обработка каждого изображения
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Обработка изображений'):
        img_path = os.path.join(DATA_DIR, 'raw', row['Image_path'])
        ground_truth = str(row['label']).strip()
        bbox_str = row['bbox']

        try:
            # Чтение и предварительная обработка изображения
            image = cv2.imread(img_path)
            if image is None:
                logging.error(f'Не удалось загрузить изображение: {img_path}')
                wers.append(1.0)
                cers.append(1.0)
                continue

            # Парсинг bbox
            bbox = parse_bbox(bbox_str)
            if not bbox:
                logging.warning(f'Пропуск изображения {img_path} из-за некорректного bbox')
                wers.append(1.0)
                cers.append(1.0)
                continue

            # Обрезка изображения по bbox
            x, y, w, h = bbox
            img_height, img_width = image.shape[:2]
            if x + w > img_width or y + h > img_height or x < 0 or y < 0 or w <= 0 or h <= 0:
                logging.warning(f'Некорректный bbox для {img_path}: {bbox}')
                wers.append(1.0)
                cers.append(1.0)
                continue
            cropped_image = image[y:y+h, x:x+w]
            cropped_image = resize_image(cropped_image, max_size=1024)

            # Извлечение текста с помощью EasyOCR
            results = reader.readtext(cropped_image, detail=0, batch_size=8, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            extracted_text = ''.join(results).strip()

            if extracted_text and ground_truth:
                wer_score = jiwer.wer(ground_truth, extracted_text)
                cer_score = jiwer.cer(ground_truth, extracted_text)
                wers.append(wer_score)
                cers.append(cer_score)
                logging.info(f'Изображение {img_path}: WER={wer_score:.4f}, CER={cer_score:.4f}, GT={ground_truth}, Pred={extracted_text}')
            else:
                logging.warning(f'Нет извлеченного текста или пустой ground truth для {img_path}')
                wers.append(1.0)
                cers.append(1.0)

            # Освобождение памяти
            del image
            del cropped_image

        except Exception as e:
            logging.error(f'Ошибка обработки {img_path}: {e}')
            wers.append(1.0)
            cers.append(1.0)

    # Вычисление средних WER и CER
    avg_wer5 = np.mean(wers) if wers else 0.0
    avg_cer5 = np.mean(cers) if cers else 0.0
    logging.info(f'Average WER: {avg_wer5:.4f}, Average CER: {avg_cer5:.4f}')

    return avg_wer5, avg_cer5

def main():
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'dataset5_test.csv')
    avg_wer5, avg_cer5 = evaluate_easyocr(csv_path)
    logging.info(f'Final Results: Average WER = {avg_wer5:.4f}, Average CER = {avg_cer5:.4f}')

if __name__ == '__main__':
    main()
