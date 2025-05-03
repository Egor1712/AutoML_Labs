# Используем официальный образ Python
FROM python:3.9

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY final_task/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем файл модели и скрипт приложения
COPY final_task/model.pkl /app/model.pkl
COPY final_task/api.py /app/api.py

# Открываем порт, на котором будет работать приложение
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
