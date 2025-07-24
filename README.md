 git clone <repository_url>
 cd Summer_practice

zip- архив с датасетами переместить в data/raw/
Переименовать: Dataset1 в Text_detection_in_the_documents
        Dataset2 в Total_text
        dataset2_1 -> sbernotes
        dataset2_2 -> russian_language_visual_text_recognition
        Dataset5 -> licance_plate_characters_detection_ocr
        Dataset6 -> ocr_machine_readable_zone_mrz_detection
     
pip install -r requirements.txt либо же копируем содержимое requirements.txt и в командной строке pip install <содержимое>
скачиваем зависимости из requirements.txt

python -m ocr_evaluate.test_total_text ->  data/processed/Total_text_results.csv  
устройство data / raw / Text_detection_in_the_documents, Total_text ... 
       
