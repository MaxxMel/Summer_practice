#!/usr/bin/env python3

"""
Оценка EasyOCR на шести OCR-датасетах

Этот скрипт загружает шесть OCR-датасетов (NT, d1, d2, d2_1, d2_2, d5, d6), выполняет оценку EasyOCR,
вычисляет метрики WER и CER, и визуализирует результаты.

Требования:
- Обработанные файлы (`test.txt`, `d1_test.csv`, `d2_test.csv`, `d2_1_test.csv`, `d2_2_test.csv`,
  `d5_test.csv`, `d6_test.csv`) должны быть в `data/processed/`.
- Сырые датасеты должны быть распакованы в `data/raw/NT/`, `data/raw/d1_raw/`, и т.д.
- Зависимости установлены через `pip install -r requirements.txt`.
- Пути в CSV (`image`, `img_path`, `Image_path`) должны быть относительными (например, `image1.jpg`)
  и указывать на файлы в `data/raw/<dataset_name>_raw/`.
"""

import pandas as pd
import easyocr
import cv2
import jiwer
import logging
import numpy as np
import os
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
from ocr_evaluation.config import PROCESSED_DATA_DIR, DATA_DIR, DATASETS

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(image, max_size=1024):
    """Изменение размера изображения с сохранением пропорций."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def parse_bbox(bbox_str, dataset_name, img_width=None, img_height=None):
    """Парсинг bbox в зависимости от формата датасета."""
    try:
        if dataset_name == 'd2_1':
            x, y, w, h = map(float, bbox_str.strip('"').split(','))
            if w <= 0 or h <= 0:
                logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
                return None
            return (int(x), int(y), int(w), int(h))
        elif dataset_name in ['d5', 'd6']:
            bbox = ast.literal_eval(bbox_str)
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
                return None
            return (int(x1), int(y1), int(w), int(h))
        elif dataset_name == 'd2_2':
            bbox = ast.literal_eval(bbox_str)
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
                return None
            return (int(x * img_width), int(y * img_height), int(w * img_width), int(h * img_height))
        return None
    except Exception as e:
        logging.error(f'Ошибка парсинга bbox: {bbox_str}, {e}')
        return None

def clean_ground_truth(text):
    """Очистка текста ground truth."""
    if pd.isna(text):
        return ''
    text = str(text).strip()
    if text.endswith(','):
        text = text[:-1]
    return text

def load_annotations(annotation_file, dataset_name):
    """Загрузка аннотаций из CSV или текстового файла."""
    annotations = {}
    if dataset_name == 'NT':
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        logging.warning(f'Некорректная строка: {line.strip()}')
                        continue
                    image_path, _, transcription = parts
                    if transcription == '###':
                        continue
                    full_image_path = os.path.join(DATA_DIR, 'raw', 'NT', os.path.basename(image_path))
                    if full_image_path not in annotations:
                        annotations[full_image_path] = []
                    annotations[full_image_path].append(transcription)
        except Exception as e:
            logging.error(f'Ошибка загрузки {annotation_file}: {e}')
            return {}
    else:
        try:
            df = pd.read_csv(annotation_file)
            df = df[df['label'].notna() & (df['label'] != '###')]
            image_column = 'image' if dataset_name in ['d1', 'd2'] else 'Image_path' if dataset_name in ['d2_2', 'd5', 'd6'] else 'img_path'
            for image_path, group in df.groupby(image_column):
                full_image_path = os.path.join(DATA_DIR, 'raw', f'{dataset_name}_raw', os.path.basename(image_path))
                annotations[full_image_path] = group['label'].tolist()
        except Exception as e:
            logging.error(f'Ошибка загрузки {annotation_file}: {e}')
            return {}
    logging.info(f'Загружено аннотаций для {len(annotations)} изображений в {dataset_name}')
    return annotations

def evaluate_dataset(dataset_name, annotation_file, reader, allowlist=None):
    """Оценка EasyOCR на одном датасете."""
    if not os.path.exists(annotation_file):
        logging.error(f'Файл аннотаций {annotation_file} не найден')
        return 0.0, 0.0

    annotations = load_annotations(annotation_file, dataset_name)
    image_paths = sorted(annotations.keys())
    logging.info(f'Найдено изображений в аннотациях: {len(image_paths)}')

    cer_scores = []
    wer_scores = []

    if dataset_name in ['NT', 'd1', 'd2']:
        for image_path in tqdm(image_paths, desc=f'Обработка {dataset_name}'):
            if not os.path.exists(image_path):
                logging.warning(f'Пропущено: {image_path} (файл не существует)')
                continue
            ground_truth = annotations.get(image_path, [])
            if not ground_truth:
                logging.warning(f'Пропущено: {image_path} (нет аннотаций)')
                continue
            try:
                results = reader.readtext(image_path, detail=0, allowlist=allowlist)
                ground_truth_text = ' '.join(ground_truth).lower()
                predicted_text = ' '.join(results).lower()
                if ground_truth_text:
                    cer_score = jiwer.cer(ground_truth_text, predicted_text)
                    wer_score = jiwer.wer(ground_truth_text, predicted_text)
                    cer_scores.append(cer_score)
                    wer_scores.append(wer_score)
                    logging.info(f'Изображение {image_path}: CER={cer_score:.4f}, WER={wer_score:.4f}')
                else:
                    logging.warning(f'Пустой ground truth для {image_path}')
                    cer_scores.append(1.0)
                    wer_scores.append(1.0)
            except Exception as e:
                logging.error(f'Ошибка обработки {image_path}: {e}')
                cer_scores.append(1.0)
                wer_scores.append(1.0)
    else:
        try:
            df = pd.read_csv(annotation_file)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Обработка {dataset_name}'):
                image_path = os.path.join(DATA_DIR, 'raw', f'{dataset_name}_raw', os.path.basename(row.get('img_path', row.get('Image_path'))))
                ground_truth = clean_ground_truth(row['label'])
                bbox_str = row['bbox']

                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        logging.error(f'Не удалось загрузить изображение: {image_path}')
                        cer_scores.append(1.0)
                        wer_scores.append(1.0)
                        continue

                    img_height, img_width = image.shape[:2]
                    bbox = parse_bbox(bbox_str, dataset_name, img_width, img_height)
                    if not bbox:
                        logging.warning(f'Пропущено: {image_path} (некорректный bbox)')
                        cer_scores.append(1.0)
                        wer_scores.append(1.0)
                        continue

                    x, y, w, h = bbox
                    if x + w > img_width or y + h > img_height or x < 0 or y < 0 or w <= 0 or h <= 0:
                        logging.warning(f'Некорректный bbox для {image_path}: {bbox}')
                        cer_scores.append(1.0)
                        wer_scores.append(1.0)
                        continue

                    cropped_image = image[y:y+h, x:x+w]
                    cropped_image = resize_image(cropped_image, max_size=128)

                    results = reader.readtext(cropped_image, detail=0, batch_size=8, allowlist=allowlist)
                    extracted_text = ''.join(results).strip() if dataset_name in ['d5', 'd6'] else ' '.join(results).strip()
                    if extracted_text and ground_truth:
                        cer_score = jiwer.cer(ground_truth, extracted_text)
                        wer_score = jiwer.wer(ground_truth, extracted_text)
                        cer_scores.append(cer_score)
                        wer_scores.append(wer_score)
                        logging.info(f'Изображение {image_path}: CER={cer_score:.4f}, WER={wer_score:.4f}, GT={ground_truth}, Pred={extracted_text}')
                    else:
                        logging.warning(f'Нет текста или пустой ground truth для {image_path}')
                        cer_scores.append(1.0)
                        wer_scores.append(1.0)

                    del cropped_image
                    del image
                except Exception as e:
                    logging.error(f'Ошибка обработки {image_path}: {e}')
                    cer_scores.append(1.0)
                    wer_scores.append(1.0)
        except Exception as e:
            logging.error(f'Ошибка чтения {annotation_file}: {e}')
            return 0.0, 0.0

    avg_cer = np.mean(cer_scores) if cer_scores else 0.0
    avg_wer = np.mean(wer_scores) if wer_scores else 0.0
    logging.info(f'{dataset_name} - Средний CER: {avg_cer:.4f}, Средний WER: {avg_wer:.4f}')

    return avg_cer, avg_wer

def main():
    """Оценка всех датасетов и визуализация результатов."""
    try:
        reader = easyocr.Reader(['en', 'ru'], gpu=True)
        logging.info('EasyOCR инициализирован с поддержкой GPU')
    except Exception as e:
        logging.warning(f'GPU не поддерживается: {e}. Используется CPU.')
        reader = easyocr.Reader(['en', 'ru'], gpu=False)

    results = {}
    for dataset_name in DATASETS:
        allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<' if dataset_name in ['d5', 'd6'] else None
        annotation_file = os.path.join(PROCESSED_DATA_DIR, 'test.txt' if dataset_name == 'NT' else f'{dataset_name}_test.csv')
        avg_cer, avg_wer = evaluate_dataset(dataset_name, annotation_file, reader, allowlist)
        results[dataset_name] = {'avg_cer': avg_cer, 'avg_wer': avg_wer}

    # Сохранение результатов
    try:
        np.savez(os.path.join(DATA_DIR, 'results.npz'), **results)
        logging.info('Результаты сохранены в data/results.npz')
    except Exception as e:
        logging.error(f'Ошибка сохранения результатов: {e}')

    # Визуализация
    if results:
        datasets = list(results.keys())
        cers = [results[ds]['avg_cer'] for ds in datasets]
        wers = [results[ds]['avg_wer'] for ds in datasets]
        x = np.arange(len(datasets))
        plt.figure(figsize=(10, 5))
        plt.bar(x - 0.2, cers, 0.4, label='CER', color='skyblue')
        plt.bar(x + 0.2, wers, 0.4, label='WER', color='salmon')
        plt.xticks(x, datasets)
        plt.xlabel('Датасет')
        plt.ylabel('Уровень ошибок')
        plt.title('Средние CER и WER по датасетам')
        plt.legend()
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(DATA_DIR, 'wer_cer_comparison.png'))
            logging.info('График сохранен в data/wer_cer_comparison.png')
        except Exception as e:
            logging.error(f'Ошибка сохранения графика: {e}')
        plt.show()
    else:
        logging.warning('Нет результатов для визуализации')

if __name__ == '__main__':
    main()
