Makefile для проекта оценки OCR

.PHONY: all install data evaluate clean

all: install data evaluate

install: @echo "Установка зависимостей..." pip install -r requirements.txt

data: @echo "Подготовка датасетов..." python -m ocr_evaluation.dataset

evaluate: @echo "Запуск оценки OCR..." python -m ocr_evaluation.modeling.predict

clean: @echo "Очистка..." rm -rf models/* rm -rf reports/figures/* rm -rf data/interim/*
