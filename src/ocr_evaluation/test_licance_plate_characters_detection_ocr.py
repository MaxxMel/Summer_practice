
#!/usr/bin/env python3

"""
Оценка EasyOCR на датасете dataset5 с сохранением результатов в CSV
"""

import pandas as pd
import easyocr
import cv2
import jiwer
import logging
import numpy as np
import os
from tqdm import tqdm
import json
import sys

# Настройка sys.path для корректного импорта при запуске из ocr_evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr_evaluation.config import DATA_DIR, PROCESSED_DATA_DIR, DATASETS
from src.ocr_evaluation.utils import resize_image, parse_bbox, clean_ground_truth

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_easyocr(csv_path):
    """Оценка EasyOCR на dataset5 с сохранением результатов в CSV."""
    if 'licance_plate_characters_detection_ocr' not in DATASETS:
        logging.error("Конфигурация для 'licance_plate_characters_detection_ocr' отсутствует в DATASETS. Проверьте config.py.")
        raise KeyError("Конфигурация для 'licance_plate_characters_detection_ocr' не найдена в DATASETS")

    config = DATASETS['licance_plate_characters_detection_ocr']
    lang = config['lang']
    allowlist = config.get('allowlist', '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # Значение по умолчанию

    try:
        reader = easyocr.Reader(lang, gpu=True)
        logging.info('EasyOCR инициализирован с поддержкой GPU')
    except Exception as e:
        logging.warning(f'GPU не поддерживается: {e}. Используется CPU.')
        reader = easyocr.Reader(lang, gpu=False)

    df = pd.read_csv(csv_path)
    logging.info(f'Загружен датасет с {len(df)} записями')

    results = []
    cer_scores = []
    wer_scores = []

    # Сохранение отладочной информации
    debug_log_path = os.path.join(DATA_DIR, 'debug_licance_plate_characters_detection_ocr.txt')
    with open(debug_log_path, 'w') as f:
        f.write('Debug log for licance_plate_characters_detection_ocr\n')

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Обработка licance_plate_characters_detection_ocr'):
        img_path = os.path.join(DATA_DIR, 'raw', row['Image_path'])
        ground_truth = clean_ground_truth(row['label'])
        bbox_str = row['bbox']

        try:
            image = cv2.imread(img_path)
            if image is None:
                logging.error(f'Не удалось загрузить изображение: {img_path}')
                results.append({
                    'image_path': img_path,
                    'ground_truth': ground_truth,
                    'predicted_text': '',
                    'cer': 1.0,
                    'wer': 1.0
                })
                cer_scores.append(1.0)
                wer_scores.append(1.0)
                with open(debug_log_path, 'a') as f:
                    f.write(f'{img_path}: Не удалось загрузить изображение\n')
                continue

            img_height, img_width = image.shape[:2]
            bbox = parse_bbox(bbox_str, dataset_name='licance_plate_characters_detection_ocr')

            # Отладка: логируем исходные значения
            logging.debug(f'Обработка {img_path}: label="{row["label"]}", ground_truth="{ground_truth}", bbox="{bbox_str}"')
            with open(debug_log_path, 'a') as f:
                f.write(f'{img_path}: label="{row["label"]}", ground_truth="{ground_truth}", bbox="{bbox_str}"\n')

            if not bbox:
                logging.warning(f'Пропуск изображения {img_path}: некорректный bbox "{bbox_str}"')
                results.append({
                    'image_path': img_path,
                    'ground_truth': ground_truth,
                    'predicted_text': '',
                    'cer': 1.0,
                    'wer': 1.0
                })
                cer_scores.append(1.0)
                wer_scores.append(1.0)
                with open(debug_log_path, 'a') as f:
                    f.write(f'{img_path}: Некорректный bbox "{bbox_str}"\n')
                continue

            if not ground_truth:
                logging.warning(f'Пропуск изображения {img_path}: пустой ground truth, bbox="{bbox_str}"')
                results.append({
                    'image_path': img_path,
                    'ground_truth': ground_truth,
                    'predicted_text': '',
                    'cer': 1.0,
                    'wer': 1.0
                })
                cer_scores.append(1.0)
                wer_scores.append(1.0)
                with open(debug_log_path, 'a') as f:
                    f.write(f'{img_path}: Пустой ground truth, bbox="{bbox_str}"\n')
                continue

            x, y, w, h = bbox
            if x + w > img_width or y + h > img_height or x < 0 or y < 0 or w <= 0 or h <= 0:
                logging.warning(f'Пропуск изображения {img_path}: некорректные размеры bbox ({x}, {y}, {w}, {h})')
                results.append({
                    'image_path': img_path,
                    'ground_truth': ground_truth,
                    'predicted_text': '',
                    'cer': 1.0,
                    'wer': 1.0
                })
                cer_scores.append(1.0)
                wer_scores.append(1.0)
                with open(debug_log_path, 'a') as f:
                    f.write(f'{img_path}: Некорректные размеры bbox ({x}, {y}, {w}, {h})\n')
                continue

            cropped_image = image[y:y+h, x:x+w]
            cropped_image = resize_image(cropped_image, max_size=1024)

            results_ocr = reader.readtext(cropped_image, detail=0, batch_size=8, allowlist=allowlist)
            predicted_text = ''.join(results_ocr).strip()

            if predicted_text and ground_truth:
                cer_score = min(jiwer.cer(ground_truth, predicted_text), 1.0)
                wer_score = min(jiwer.wer(ground_truth, predicted_text), 1.5)
                cer_scores.append(cer_score)
                wer_scores.append(wer_score)
                logging.info(f'Изображение {img_path}: CER={cer_score:.4f}, WER={wer_score:.4f}, GT={ground_truth}, Pred={predicted_text}')
            else:
                logging.warning(f'Нет извлеченного текста или пустой ground truth для {img_path}, bbox={bbox}')
                cer_score = 1.0
                wer_score = 1.0
                cer_scores.append(cer_score)
                wer_scores.append(wer_score)
                with open(debug_log_path, 'a') as f:
                    f.write(f'{img_path}, bbox={bbox}: GT={ground_truth}, Pred={predicted_text}\n')

            results.append({
                'image_path': img_path,
                'ground_truth': ground_truth,
                'predicted_text': predicted_text,
                'cer': cer_score,
                'wer': wer_score
            })

            del image
            del cropped_image

        except Exception as e:
            logging.error(f'Ошибка обработки {img_path}: {e}')
            results.append({
                'image_path': img_path,
                'ground_truth': ground_truth,
                'predicted_text': '',
                'cer': 1.0,
                'wer': 1.0
            })
            cer_scores.append(1.0)
            wer_scores.append(1.0)
            with open(debug_log_path, 'a') as f:
                f.write(f'{img_path}: Ошибка обработки - {e}\n')

    avg_cer = np.mean(cer_scores) if cer_scores else 0.0
    avg_wer = np.mean(wer_scores) if wer_scores else 0.0
    logging.info(f'dataset5 - Средний CER: {avg_cer:.4f}, Средний WER: {avg_wer:.4f}')

    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    csv_output_path = os.path.join(PROCESSED_DATA_DIR, 'results_licance_plate_characters_detection_ocr_.csv')
    results_df.to_csv(csv_output_path, index=False)
    logging.info(f'Таблица результатов сохранена в {csv_output_path}')

    # Сохранение средних значений в JSON
    json_results = {'avg_cer': avg_cer, 'avg_wer': avg_wer}
    json_output_path = os.path.join(DATA_DIR, 'ocr_results_licance_plate_characters_detection_ocr.json')
    with open(json_output_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    logging.info(f'Результаты сохранены в {json_output_path}')

    logging.info(f'Отладочная информация сохранена в {debug_log_path}')
    return avg_cer, avg_wer

def main():
    """Оценка EasyOCR на dataset5."""
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'licance_plate_characters_detection_ocr_test.csv')
    avg_cer, avg_wer = evaluate_easyocr(csv_path)
    logging.info(f'Final Results: Average CER = {avg_cer:.4f}, Average WER = {avg_wer:.4f}')

if __name__ == '__main__':
    main()
