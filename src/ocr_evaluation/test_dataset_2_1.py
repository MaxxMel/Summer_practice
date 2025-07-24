
#!/usr/bin/env python3

"""
Оценка EasyOCR на датасете dataset2_1 с сохранением результатов в CSV
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
import gc
import psutil
import sys

# Настройка sys.path для корректного импорта при запуске из ocr_evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr_evaluation.config import DATA_DIR, PROCESSED_DATA_DIR, DATASETS
from src.ocr_evaluation.utils import resize_image, parse_bbox, clean_ground_truth

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_easyocr(csv_path, chunk_size=50, save_interval=10):
    """Оценка EasyOCR на dataset2_1 с сохранением результатов в CSV."""
    config = DATASETS['dataset2_1']
    lang = config['lang']
    allowlist = config['allowlist']
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

    grouped = df.groupby('img_path')
    total_images = len(grouped)
    logging.info(f'Найдено уникальных изображений: {total_images}')

    # Сохранение отладочной информации о проблемных записях
    debug_log_path = os.path.join(DATA_DIR, 'debug_dataset2_1.txt')
    with open(debug_log_path, 'w') as f:
        f.write('Debug log for dataset2_1\n')

    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        chunk_groups = list(grouped)[chunk_start:chunk_end]

        for img_idx, (img_path, group) in enumerate(tqdm(chunk_groups, desc=f'Обработка dataset2_1 (чанк {chunk_start//chunk_size + 1})')):
            try:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                logging.info(f'Использование памяти перед изображением {img_path}: {mem_info.rss / 1024**2:.2f} MB')

                full_image_path = os.path.join(DATA_DIR, 'raw', img_path)
                image = cv2.imread(full_image_path)
                if image is None:
                    logging.error(f'Не удалось загрузить изображение: {full_image_path}')
                    results.extend([{
                        'image_path': full_image_path,
                        'ground_truth': clean_ground_truth(row['label']),
                        'predicted_text': '',
                        'cer': 1.0,
                        'wer': 1.0
                    } for _, row in group.iterrows()])
                    cer_scores.extend([1.0] * len(group))
                    wer_scores.extend([1.0] * len(group))
                    with open(debug_log_path, 'a') as f:
                        f.write(f'{full_image_path}: Не удалось загрузить изображение\n')
                    continue

                img_height, img_width = image.shape[:2]
                image = resize_image(image, max_size=512)
                resized_height, resized_width = image.shape[:2]
                scale_x = resized_width / img_width
                scale_y = resized_height / img_height

                for _, row in group.iterrows():
                    ground_truth = clean_ground_truth(row['label'])
                    bbox_str = row['bbox']
                    bbox = parse_bbox(bbox_str, dataset_name='dataset2_1')

                    # Отладка: логируем исходные значения
                    logging.debug(f'Обработка {full_image_path}: label="{row["label"]}", ground_truth="{ground_truth}", bbox="{bbox_str}"')
                    with open(debug_log_path, 'a') as f:
                        f.write(f'{full_image_path}: label="{row["label"]}", ground_truth="{ground_truth}", bbox="{bbox_str}"\n')

                    if not bbox:
                        logging.warning(f'Пропуск bbox для {full_image_path}: некорректный bbox "{bbox_str}"')
                        results.append({
                            'image_path': full_image_path,
                            'ground_truth': ground_truth,
                            'predicted_text': '',
                            'cer': 1.0,
                            'wer': 1.0
                        })
                        cer_scores.append(1.0)
                        wer_scores.append(1.0)
                        continue

                    if not ground_truth:
                        logging.warning(f'Пропуск bbox для {full_image_path}: пустой ground truth, bbox="{bbox_str}"')
                        results.append({
                            'image_path': full_image_path,
                            'ground_truth': ground_truth,
                            'predicted_text': '',
                            'cer': 1.0,
                            'wer': 1.0
                        })
                        cer_scores.append(1.0)
                        wer_scores.append(1.0)
                        continue

                    x, y, w, h = bbox
                    x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

                    if x + w > resized_width or y + h > resized_height or x < 0 or y < 0 or w <= 0 or h <= 0:
                        logging.warning(f'Пропуск bbox для {full_image_path}: некорректные размеры bbox ({x}, {y}, {w}, {h})')
                        results.append({
                            'image_path': full_image_path,
                            'ground_truth': ground_truth,
                            'predicted_text': '',
                            'cer': 1.0,
                            'wer': 1.0
                        })
                        cer_scores.append(1.0)
                        wer_scores.append(1.0)
                        with open(debug_log_path, 'a') as f:
                            f.write(f'{full_image_path}: некорректные размеры bbox ({x}, {y}, {w}, {h})\n')
                        continue

                    cropped_image = image[y:y+h, x:x+w]
                    if max(cropped_image.shape[:2]) > 128:
                        cropped_image = resize_image(cropped_image, max_size=128)

                    results_ocr = reader.readtext(cropped_image, detail=0, batch_size=2, allowlist=allowlist)
                    predicted_text = ' '.join(results_ocr).strip()

                    if predicted_text and ground_truth:
                        cer_score = min(jiwer.cer(ground_truth, predicted_text), 1.0)
                        wer_score = min(jiwer.wer(ground_truth, predicted_text), 1.5)
                        cer_scores.append(cer_score)
                        wer_scores.append(wer_score)
                        logging.info(f'Изображение {full_image_path}, bbox {bbox}: CER={cer_score:.4f}, WER={wer_score:.4f}, GT={ground_truth}, Pred={predicted_text}')
                    else:
                        logging.warning(f'Нет извлеченного текста или пустой ground truth для {full_image_path}, bbox {bbox}')
                        cer_score = 1.0
                        wer_score = 1.0
                        cer_scores.append(cer_score)
                        wer_scores.append(wer_score)
                        with open(debug_log_path, 'a') as f:
                            f.write(f'{full_image_path}, bbox {bbox}: GT={ground_truth}, Pred={predicted_text}\n')

                    results.append({
                        'image_path': full_image_path,
                        'ground_truth': ground_truth,
                        'predicted_text': predicted_text,
                        'cer': cer_score,
                        'wer': wer_score
                    })

                    del cropped_image
                    gc.collect()

                del image
                gc.collect()

                if (img_idx + 1) % save_interval == 0:
                    np.savez(os.path.join(DATA_DIR, f'intermediate_results_dataset2_1_chunk_{chunk_start//chunk_size + 1}.npz'), wers=wer_scores, cers=cer_scores)
                    logging.info(f'Сохранены промежуточные результаты для чанка {chunk_start//chunk_size + 1}')

            except Exception as e:
                logging.error(f'Ошибка обработки {full_image_path}: {e}')
                results.extend([{
                    'image_path': full_image_path,
                    'ground_truth': clean_ground_truth(row['label']),
                    'predicted_text': '',
                    'cer': 1.0,
                    'wer': 1.0
                } for _, row in group.iterrows()])
                cer_scores.extend([1.0] * len(group))
                wer_scores.extend([1.0] * len(group))
                with open(debug_log_path, 'a') as f:
                    f.write(f'{full_image_path}: Ошибка обработки - {e}\n')

    avg_cer = np.mean(cer_scores) if cer_scores else 0.0
    avg_wer = np.mean(wer_scores) if wer_scores else 0.0
    logging.info(f'dataset2_1 - Средний CER: {avg_cer:.4f}, Средний WER: {avg_wer:.4f}')

    # Сохранение результатов в CSV
    results_df = pd.DataFrame(results)
    csv_output_path = os.path.join(PROCESSED_DATA_DIR, 'results_dataset2_1.csv')
    results_df.to_csv(csv_output_path, index=False)
    logging.info(f'Таблица результатов сохранена в {csv_output_path}')

    # Сохранение средних значений в JSON
    json_results = {'avg_cer': avg_cer, 'avg_wer': avg_wer}
    json_output_path = os.path.join(DATA_DIR, 'ocr_results_dataset2_1.json')
    with open(json_output_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    logging.info(f'Результаты сохранены в {json_output_path}')

    # Сохранение финальных результатов в NPZ
    np.savez(os.path.join(DATA_DIR, 'final_results_dataset2_1.npz'), wers=wer_scores, cers=cer_scores, avg_wer=avg_wer, avg_cer=avg_cer)
    logging.info(f'Финальные результаты сохранены в {os.path.join(DATA_DIR, "final_results_dataset2_1.npz")}')

    logging.info(f'Отладочная информация сохранена в {debug_log_path}')
    return avg_cer, avg_wer

def main():
    """Оценка EasyOCR на dataset2_1."""
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'dataset2_1_test.csv')
    avg_cer, avg_wer = evaluate_easyocr(csv_path)
    logging.info(f'Final Results: Average CER = {avg_cer:.4f}, Average WER = {avg_wer:.4f}')

if __name__ == '__main__':
    main()
