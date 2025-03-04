#!/bin/bash
# Скрипт pipeline.sh выполняет следующие шаги:
# 1. Устанавливает необходимые пакеты из requirements.txt.
# 2. Запускает по очереди:
#    - data_creation.py
#    - data_preprocessing.py
#    - model_preparation.py
#    - model_testing.py
#
# При успешном выполнении всех этапов в стандартный поток вывода выводится итоговая метрика модели

# Функция для проверки статуса выполнения предыдущей команды
check_status() {
    if [ $? -ne 0 ]; then
        echo "Ошибка при выполнении $1" >&2
        exit 1
    fi
}

echo "Установка необходимых пакетов из requirements.txt..."
pip install -r requirements.txt
check_status "pip install -r requirements.txt"

echo "Запуск data_creation.py..."
python data_creation.py
check_status "data_creation.py"

echo "Запуск data_preprocessing.py..."
python data_preprocessing.py
check_status "data_preprocessing.py"

echo "Запуск model_preparation.py..."
python model_preparation.py
check_status "model_preparation.py"

echo "Запуск model_testing.py..."
# Выполняем model_testing.py и извлекаем последнюю строку вывода (предполагается, что это строка с итоговой метрикой)
metric=$(python model_testing.py | tail -n 1)
check_status "model_testing.py"

# Вывод итоговой метрики в стандартный поток вывода
echo "$metric"
