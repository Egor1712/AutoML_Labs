import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_training_temperatures(train_folder):
    """
    Загружает столбец 'Temperature' из всех CSV файлов в папке train.
    
    Аргументы:
    train_folder (str): путь к папке с тренировочными данными.
    
    Возвращает:
    np.ndarray: объединённый массив значений температуры для обучения StandardScaler.
    """
    csv_files = glob.glob(os.path.join(train_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"В папке '{train_folder}' не найдено CSV файлов.")
    temperatures = []
    for file in csv_files:
        df = pd.read_csv(file)
        if "Temperature" not in df.columns:
            raise ValueError(f"Файл {file} не содержит колонки 'Temperature'.")
        temperatures.append(df["Temperature"].values.reshape(-1, 1))
    return np.vstack(temperatures)

def preprocess_and_save_data(folder, scaler, output_folder):
    """
    Применяет StandardScaler к данным из CSV файлов в указанной папке и сохраняет преобразованные данные.
    
    Аргументы:
    folder (str): папка с исходными CSV файлами.
    scaler (StandardScaler): обученный объект StandardScaler.
    output_folder (str): папка для сохранения преобразованных файлов.
    """
    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    for file in csv_files:
        df = pd.read_csv(file)
        # Применяем StandardScaler к колонке 'Temperature'
        df["Temperature"] = scaler.transform(df[["Temperature"]])
        output_file = os.path.join(output_folder, os.path.basename(file))
        df.to_csv(output_file, index=False)
        print(f"Файл сохранён: {output_file}")

def main():
    train_folder = "train"
    test_folder = "test"
    train_output_folder = "train_preprocessed"
    test_output_folder = "test_preprocessed"
    
    # Обучаем StandardScaler на данных тренировочной выборки
    X_train = load_training_temperatures(train_folder)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Применяем трансформацию и сохраняем преобразованные данные для обеих выборок
    preprocess_and_save_data(train_folder, scaler, train_output_folder)
    preprocess_and_save_data(test_folder, scaler, test_output_folder)

if __name__ == "__main__":
    main()
