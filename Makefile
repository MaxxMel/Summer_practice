install: pip install -r requirements.txt pip install -e .

data: @echo "Extract dataset zip archives to data/raw/"

test_text_detection: python -m src.ocr_evaluation.test_total_text

test_ocr_machine_readable_zone_mrz_detection: python -m src.ocr_evaluation.test_ocr_machine_readable_zone_mrz_detection

clean: rm -rf pycache src/ocr_evaluation/pycache

.PHONY: install data test_text_detection test_dataset6 clean
