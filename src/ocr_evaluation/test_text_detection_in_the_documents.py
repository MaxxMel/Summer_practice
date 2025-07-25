
#!/usr/bin/env python3

"""
Оценка EasyOCR на датасете Dataset1 с сохранением результатов в CSV
"""

import pandas as pd
import easyocr
import jiwer
import logging
import numpy as np
import os
from tqdm import tqdm
import json
from src.ocr_evaluation.config import DATA_DIR, PROCESSED_DATA_DIR, DATASETS
from src.ocr_evaluation.utils import resize_image, clean_ground_truth

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_annotations(annotation_file):
    """Загрузка аннотаций из CSV."""
    annotations = {}
    try:
        df = pd.read_csv(annotation_file)
        df = df[df['label'].notna() & (df['label'] != '###')]
        logging.info(f'Загружен CSV {annotation_file} с {len(df)} строками')
        for image_path, group in df.groupby('image'):
            full_image_path = os.path.join(DATA_DIR, 'raw', image_path)
            annotations[full_image_path] = [clean_ground_truth(label) for label in group['label'].tolist()]
        logging.info(f'Загружено аннотаций для {len(annotations)} изображений')
        logging.info(f'Примеры путей: {list(annotations.keys())[:5]}')
        return annotations
    except Exception as e:
        logging.error(f'Ошибка загрузки {annotation_file}: {e}')
        return {}

def evaluate_easyocr(dataset_path, annotation_file, num_images=None):
    """Оценка EasyOCR на text_detection_in_the_documents с сохранением результатов в CSV."""
    config = DATASETS['Text_detection_in_the_documents']
    lang = config['lang']
    try:
        reader = easyocr.Reader(lang, gpu=True)
        logging.info('EasyOCR инициализирован с поддержкой GPU')
    except Exception as e:
        logging.warning(f'GPU не поддерживается: {e}. Используется CPU.')
        reader = easyocr.Reader(['en'], gpu=False)

    annotations = load_annotations(annotation_file)
    image_paths = sorted(annotations.keys())
    logging.info(f'Найдено изображений в аннотациях: {len(image_paths)}')

    if num_images is not None:
        image_paths = image_paths[:num_images]

    results = []
    cer_scores = []
    wer_scores = []

    for image_path in tqdm(image_paths, desc="Обработка Text_detection_in_the_documents"):
        if not os.path.exists(image_path):
            logging.warning(f'Пропущено: {image_path} (файл не существует)')
            results.append({
                'image_path': image_path,
                'ground_truth': '',
                'predicted_text': '',
                'cer': 1.0,
                'wer': 1.0
            })
            cer_scores.append(1.0)
            wer_scores.append(1.0)
            continue

        ground_truth = annotations.get(image_path, [])
        if not ground_truth:
            logging.warning(f'Пропущено: {image_path} (нет аннотаций)')
            results.append({
                'image_path': image_path,
                'ground_truth': '',
                'predicted_text': '',
                'cer': 1.0,
                'wer': 1.0
            })
            cer_scores.append(1.0)
            wer_scores.append(1.0)
            continue

        try:
            results_ocr = reader.readtext(image_path, detail=0)
            ground_truth_text = ' '.join(ground_truth).lower()
            predicted_text = ' '.join(results_ocr).lower()
            if ground_truth_text:
                cer_score = min(jiwer.cer(ground_truth_text, predicted_text), 1.0)
                wer_score = min(jiwer.wer(ground_truth_text, predicted_text), 1.5)
                cer_scores.append(cer_score)
                wer_scores.append(wer_score)
                logging.info(f'Изображение {image_path}: CER={cer_score:.4f}, WER={wer_score:.4f}, GT={ground_truth_text}, Pred={predicted_text}')
            else:
                logging.warning(f'Пустой ground truth для {image_path}')
                cer_score = 1.0
                wer_score = 1.0
                cer_scores.append(cer_score)
                wer_scores.append(wer_score)

            results.append({
                'image_path': image_path,
                'ground_truth': ground_truth_text,
                'predicted_text': predicted_text,
                'cer': cer_score,
                'wer': wer_score
            })

        except Exception as e:
            logging.error(f'Ошибка обработки {image_path}: {e}')
            results.append({
                'image_path': image_path,
                'ground_truth': ground_truth_text,
                'predicted_text': '',
                'cer': 1.0,
                'wer': 1.0
            })
            cer_scores.append(1.0)
            wer_scores.append(1.0)

    avg_cer = np.mean(cer_scores) if cer_scores else 0.0
    avg_wer = np.mean(wer_scores) if wer_scores else 0.0
    logging.info(f'Dataset1 - Средний CER: {avg_cer:.4f}, Средний WER: {avg_wer:.4f}')

    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    csv_output_path = os.path.join(PROCESSED_DATA_DIR, 'results_text_detection_in_the_documents.csv')
    results_df.to_csv(csv_output_path, index=False)
    logging.info(f'Таблица результатов сохранена в {csv_output_path}')

    # Сохранение средних значений в JSON
    json_results = {'avg_cer': avg_cer, 'avg_wer': avg_wer}
    json_output_path = os.path.join(dataset_path, 'ocr_results_text_detection_in_the_documents.json')
    with open(json_output_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    logging.info(f'Результаты сохранены в {json_output_path}')

    return avg_cer, avg_wer

def main():
    """Оценка EasyOCR на Dataset1."""
    dataset_path = DATA_DIR
    annotation_file = os.path.join(PROCESSED_DATA_DIR, 'text_detection_in_the_documents_test.csv')
    avg_cer, avg_wer = evaluate_easyocr(dataset_path, annotation_file)
    logging.info(f'Final Results: Average CER = {avg_cer:.4f}, Average WER = {avg_wer:.4f}')

if __name__ == '__main__':
    main()

#python -m ocr_evaluation.test_dataset1
