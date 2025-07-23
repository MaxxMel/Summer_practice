
#!/usr/bin/env python3

import pandas as pd
import easyocr
import cv2
import jiwer
import logging
import numpy as np
import os
from tqdm import tqdm
import gc
import psutil
from ocr_evaluation.config import DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resize_image(image, max_size=512):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def parse_bbox(bbox_str):
    try:
        x, y, w, h = map(float, bbox_str.strip('"').split(','))
        if w <= 0 or h <= 0:
            logging.warning(f'Некорректные размеры bbox: w={w}, h={h}')
            return None
        return (int(x), int(y), int(w), int(h))
    except Exception as e:
        logging.error(f'Ошибка парсинга bbox: {bbox_str}, {e}')
        return None

def clean_ground_truth(text):
    if pd.isna(text):
        return ''
    text = str(text).strip()
    if text.endswith(','):
        text = text[:-1]
    return text

def evaluate_easyocr(csv_path, chunk_size=50, save_interval=10):
    try:
        reader = easyocr.Reader(['ru'], gpu=True)
        logging.info('EasyOCR инициализирован с поддержкой GPU')
    except Exception as e:
        logging.warning(f'GPU не поддерживается: {e}. Используется CPU.')
        reader = easyocr.Reader(['ru'], gpu=False)

    allowlist = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя '


    df = pd.read_csv(csv_path)
    logging.info(f'Загружен датасет с {len(df)} записями')

    wers = []
    cers = []

    grouped = df.groupby('img_path')
    total_images = len(grouped)
    logging.info(f'Найдено уникальных изображений: {total_images}')

    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        chunk_groups = list(grouped)[chunk_start:chunk_end]

        for img_idx, (img_path, group) in enumerate(tqdm(chunk_groups, desc=f'Обработка чанка {chunk_start//chunk_size + 1}')):
            try:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                logging.info(f'Использование памяти перед изображением {img_path}: {mem_info.rss / 1024**2:.2f} MB')

                full_image_path = os.path.join(DATA_DIR, 'raw', img_path)
                image = cv2.imread(full_image_path)
                if image is None:
                    logging.error(f'Не удалось загрузить изображение: {full_image_path}')
                    wers.extend([1.0] * len(group))
                    cers.extend([1.0] * len(group))
                    continue

                img_height, img_width = image.shape[:2]
                image = resize_image(image, max_size=512)
                resized_height, resized_width = image.shape[:2]
                scale_x = resized_width / img_width
                scale_y = resized_height / img_height

                for _, row in group.iterrows():
                    ground_truth = clean_ground_truth(row['label'])
                    bbox = parse_bbox(row['bbox'])

                    if not bbox or not ground_truth:
                        logging.warning(f'Пропуск bbox для {full_image_path} из-за некорректного bbox или пустого ground truth')
                        wers.append(1.0)
                        cers.append(1.0)
                        continue

                    x, y, w, h = bbox
                    x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

                    if x + w > resized_width or y + h > resized_height or x < 0 or y < 0 or w <= 0 or h <= 0:
                        logging.warning(f'Некорректный bbox для {full_image_path}: ({x}, {y}, {w}, {h})')
                        wers.append(1.0)
                        cers.append(1.0)
                        continue

                    cropped_image = image[y:y+h, x:x+w]

                    if max(cropped_image.shape[:2]) > 128:
                        cropped_image = resize_image(cropped_image, max_size=128)

                    results = reader.readtext(cropped_image, detail=0, batch_size=2, allowlist=allowlist)
                    extracted_text = ' '.join(results).strip()

                    if extracted_text and ground_truth:
                        wer_score = jiwer.wer(ground_truth, extracted_text)
                        cer_score = jiwer.cer(ground_truth, extracted_text)
                        wers.append(wer_score)
                        cers.append(cer_score)
                        logging.info(f'Изображение {full_image_path}, bbox {bbox}: WER={wer_score:.4f}, CER={cer_score:.4f}, GT={ground_truth}, Pred={extracted_text}')
                    else:
                        logging.warning(f'Нет извлеченного текста или пустой ground truth для {full_image_path}, bbox {bbox}')
                        wers.append(1.0)
                        cers.append(1.0)

                    del cropped_image
                    gc.collect()

                del image
                gc.collect()

                if (img_idx + 1) % save_interval == 0:
                    np.savez(os.path.join(DATA_DIR, f'intermediate_results_chunk_{chunk_start//chunk_size + 1}.npz'), wers=wers, cers=cers)
                    logging.info(f'Сохранены промежуточные результаты для чанка {chunk_start//chunk_size + 1}')

            except Exception as e:
                logging.error(f'Ошибка обработки {full_image_path}: {e}')
                wers.extend([1.0] * len(group))
                cers.extend([1.0] * len(group))

    avg_wer2_1 = np.mean(wers) if wers else 0.0
    avg_cer2_1 = np.mean(cers) if cers else 0.0
    logging.info(f'Average WER: {avg_wer2_1:.4f}, Average CER: {avg_cer2_1:.4f}')

    np.savez(os.path.join(DATA_DIR, 'final_results_d2_1.npz'), wers=wers, cers=cers, avg_wer=avg_wer2_1, avg_cer=avg_cer2_1)
    logging.info(f'Финальные результаты сохранены в {os.path.join(DATA_DIR, "final_results_d2_1.npz")}')

    return avg_wer2_1, avg_cer2_1

def main():
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'dataset2_1_test.csv')
    avg_wer2_1, avg_cer2_1 = evaluate_easyocr(csv_path)
    logging.info(f'Final Results: Average WER = {avg_wer2_1:.4f}, Average CER = {avg_cer2_1:.4f}')

if __name__ == '__main__':
    main()
