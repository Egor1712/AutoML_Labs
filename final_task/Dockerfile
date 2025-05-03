# Используем официальный образ Python
FROM python:3.9

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY lab_1/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем файл модели и скрипт приложения
COPY lab_1/model.pkl /app/model.pkl
COPY lab_3/api.py /app/api.py

# Открываем порт, на котором будет работать приложение
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
