#!/usr/bin/env python3

import pandas as pd
import easyocr
import jiwer
import logging
import numpy as np
import os
from tqdm import tqdm
import json
from ocr_evaluation.config import DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_annotations(annotation_file):
    annotations = {}
    try:
        df = pd.read_csv(annotation_file)
        df = df[df['label'].notna() & (df['label'] != '###')]
        logging.info(f'Загружен CSV {annotation_file} с {len(df)} строками')
        for image_path, group in df.groupby('Image_path'):
            full_image_path = os.path.join(DATA_DIR, 'raw', image_path)
            logging.info(f'Сформирован путь к изображению: {full_image_path}')
            annotations[full_image_path] = group['label'].tolist()
        logging.info(f'Загружено аннотаций для {len(annotations)} изображений')
        logging.info(f'Примеры путей: {list(annotations.keys())[:5]}')
        return annotations
    except Exception as e:
        logging.error(f'Ошибка загрузки {annotation_file}: {e}')
        return {}

def evaluate_easyocr(dataset_path, annotation_file, num_images=None):
    try:
        reader = easyocr.Reader(['en', 'ru'], gpu=True)
        logging.info('EasyOCR инициализирован с поддержкой GPU')
    except Exception as e:
        logging.warning(f'GPU не поддерживается: {e}. Используется CPU.')
        reader = easyocr.Reader(['en', 'ru'], gpu=False)

    annotations = load_annotations(annotation_file)
    image_paths = sorted(annotations.keys())
    logging.info(f'Найдено изображений в аннотациях: {len(image_paths)}')

    if num_images is not None:
        image_paths = image_paths[:num_images]

    cer_scores = []
    wer_scores = []

    for image_path in tqdm(image_paths, desc="Обработка Dataset2"):
        if not os.path.exists(image_path):
            logging.warning(f'Пропущено: {image_path} (файл не существует)')
            cer_scores.append(1.0)
            wer_scores.append(1.0)
            continue

        ground_truth = annotations.get(image_path, [])
        if not ground_truth:
            logging.warning(f'Пропущено: {image_path} (нет аннотаций)')
            cer_scores.append(1.0)
            wer_scores.append(1.0)
            continue

        try:
            results = reader.readtext(image_path, detail=0)
            ground_truth_text = ' '.join(ground_truth).lower()
            predicted_text = ' '.join(results).lower()
            if ground_truth_text:
                cer_score = jiwer.cer(ground_truth_text, predicted_text)
                wer_score = jiwer.wer(ground_truth_text, predicted_text)
                cer_scores.append(cer_score)
                wer_scores.append(wer_score)
                logging.info(f'Изображение {image_path}: CER={cer_score:.4f}, WER={wer_score:.4f}, GT={ground_truth_text}, Pred={predicted_text}')
            else:
                logging.warning(f'Пустой ground truth для {image_path}')
                cer_scores.append(1.0)
                wer_scores.append(1.0)
        except Exception as e:
            logging.error(f'Ошибка обработки {image_path}: {e}')
            cer_scores.append(1.0)
            wer_scores.append(1.0)

    avg_cer2 = np.mean(cer_scores) if cer_scores else 0.0
    avg_wer2 = np.mean(wer_scores) if wer_scores else 0.0
    logging.info(f'Dataset2 - Средний CER: {avg_cer2:.4f}, Средний WER: {avg_wer2:.4f}')

    results = {'avg_cer2': avg_cer2, 'avg_wer2': avg_wer2}
    with open(os.path.join(dataset_path, 'ocr_results2.json'), 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f'Результаты сохранены в {os.path.join(dataset_path, "ocr_results2.json")}')

    return avg_cer2, avg_wer2

def main():
    dataset_path = DATA_DIR 
    annotation_file = os.path.join(PROCESSED_DATA_DIR, 'dataset2_test.csv')

    avg_cer2, avg_wer2 = evaluate_easyocr(dataset_path, annotation_file)
    logging.info(f'Average CER: {avg_cer2:.4f}')
    logging.info(f'Average WER: {avg_wer2:.4f}')

if __name__ == '__main__':
    main()
