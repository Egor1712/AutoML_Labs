#!/bin/bash

echo "1. Инициализация Git и DVC"
git init
dvc init

echo "2. Настройка удалённого хранилища DVC"
FOLDER_ID="1sOUGzlBSV5XNit7YW0WXiAw5RLj0-qX7"

dvc remote add -d myremote gdrive://$FOLDER_ID
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path credentials.json

echo "Удалённое хранилище настроено"

echo "3. Установка зависимостей"
pip install -r requirements.txt

echo "4. Запуск пайплайна"
dvc repro

echo "5. Отправка данных в облако"
dvc push

echo "6. Фиксация изменений"
git add .
git commit -m "DVC Pipeline"