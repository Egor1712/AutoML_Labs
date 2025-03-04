import os
import numpy as np
import pandas as pd

def generate_dataset(days=30, noise_level=1.0, add_anomaly=False):
    """
    Генерирует набор данных с температурой за заданное количество дней.
    
    Аргументы:
    days (int): количество дней (по умолчанию 30).
    noise_level (float): стандартное отклонение для добавления случайного шума.
    add_anomaly (bool): если True, в случайный день добавляется аномальное значение.
    
    Возвращает:
    pd.DataFrame с колонками "Day" (номер дня) и "Temperature" (значение температуры).
    """
    # Массив дней
    days_array = np.arange(1, days + 1)
    # Базовая температура с синусоидальной моделью (имитация сезонности)
    base_temp = 20
    amplitude = 10
    temperature = base_temp + amplitude * np.sin(2 * np.pi * days_array / days)
    
    # Добавляем шум
    temperature += np.random.normal(0, noise_level, size=days)
    
    # Вставляем аномалию, если требуется
    if add_anomaly:
        anomaly_index = np.random.randint(0, days)
        # Аномалия: резкое увеличение или падение температуры (умножаем на случайный коэффициент)
        factor = np.random.choice([0.5, 1.5])
        temperature[anomaly_index] *= factor
    
    # Формируем DataFrame
    df = pd.DataFrame({
        "Day": days_array,
        "Temperature": temperature
    })
    return df

def main():
    # Создаем директории для тренировочных и тестовых наборов данных
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    
    total_datasets = 6  # Общее количество наборов данных
    
    for i in range(total_datasets):
        # Чередуем сохранение: четные наборы в "train", нечетные – в "test"
        folder = "train" if i % 2 == 0 else "test"
        
        # Задаем уровень шума и случайно решаем, добавлять ли аномалию
        noise_level = np.random.uniform(0.5, 2.0)
        add_anomaly = np.random.choice([True, False])
        
        # Генерация набора данных
        df = generate_dataset(days=30, noise_level=noise_level, add_anomaly=add_anomaly)
        filename = os.path.join(folder, f"dataset_{i+1}.csv")
        df.to_csv(filename, index=False)
        print(f"Набор данных {i+1} сохранен в {filename}")

if __name__ == "__main__":
    main()
