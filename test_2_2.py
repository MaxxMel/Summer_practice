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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def parse_bbox(bbox_str, img_width, img_height):
    try:
        bbox = ast.literal_eval(bbox_str)
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
            return None
        x = int(x * img_width)
        y = int(y * img_height)
        w = int(w * img_width)
        h = int(h * img_height)
        return (x, y, w, h)
    except Exception as e:
        logging.error(f'Ошибка парсинга bbox: {bbox_str}, {e}')
        return None

def evaluate_easyocr(csv_path):
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        logging.info('EasyOCR инициализирован с поддержкой GPU')
    except Exception as e:
        logging.warning(f'GPU не поддерживается: {e}. Используется CPU.')
        reader = easyocr.Reader(['en'], gpu=False)

    df = pd.read_csv(csv_path)
    logging.info(f'Загружен датасет с {len(df)} записями')

    wers = []
    cers = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Обработка изображений'):
        img_path = os.path.join(DATA_DIR, 'raw', row['image_path'])
        ground_truth = row['label']
        bbox_str = row['bbox']

        try:
            image = cv2.imread(img_path)
            if image is None:
                logging.error(f'Не удалось загрузить изображение: {img_path}')
                wers.append(1.0)
                cers.append(1.0)
                continue

            img_height, img_width = image.shape[:2]
            bbox = parse_bbox(bbox_str, img_width, img_height)
            if not bbox:
                logging.warning(f'Пропуск изображения {img_path} из-за некорректного bbox')
                wers.append(1.0)
                cers.append(1.0)
                continue

            x, y, w, h = bbox
            if x + w > img_width or y + h > img_height or x < 0 or y < 0 or w <= 0 or h <= 0:
                logging.warning(f'Некорректный bbox для {img_path}: {bbox}')
                wers.append(1.0)
                cers.append(1.0)
                continue
            cropped_image = image[y:y+h, x:x+w]
            cropped_image = resize_image(cropped_image, max_size=1024)

            results = reader.readtext(cropped_image, detail=0, batch_size=8)
            extracted_text = ' '.join(results).strip()

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


            del image
            del cropped_image

        except Exception as e:
            logging.error(f'Ошибка обработки {img_path}: {e}')
            wers.append(1.0)
            cers.append(1.0)

    avg_wer2_2 = np.mean(wers) if wers else 0.0
    avg_cer2_2 = np.mean(cers) if cers else 0.0
    logging.info(f'Average WER: {avg_wer2_2:.4f}, Average CER: {avg_cer2_2:.4f}')

    return avg_wer2_2, avg_cer2_2

def main():
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'dataset2_2_test.csv')
    avg_wer2_2, avg_cer2_2 = evaluate_easyocr(csv_path)
    logging.info(f'Final Results: Average WER = {avg_wer2_2:.4f}, Average CER = {avg_cer2_2:.4f}')

if __name__ == '__main__':
    main()
