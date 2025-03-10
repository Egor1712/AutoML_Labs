import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_training_temperatures():
    """
    Загружает столбец 'Temperature' из всех CSV файлов в папке train.
    
    Аргументы:
    train_folder (str): путь к папке с тренировочными данными.
    
    Возвращает:
    np.ndarray: объединённый массив значений температуры для обучения StandardScaler.
    """
    df = pd.read_csv('./dataset.csv')
    return df

def preprocess_and_save_data(data, output_folder):
    """
    Применяет StandardScaler к данным из CSV файлов в указанной папке и сохраняет преобразованные данные.
    
    Аргументы:
    folder (str): папка с исходными CSV файлами.
    scaler (StandardScaler): обученный объект StandardScaler.
    output_folder (str): папка для сохранения преобразованных файлов.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename('dataset.csv'))
    data.to_csv(output_file, index=False)
    print(f"Файл сохранён: {output_file}")

def main():
    train_output_folder = "train_preprocessed"
    test_output_folder = "test_preprocessed"
    
    data = load_training_temperatures()
    scaler = StandardScaler()
    data[["Temperature"]] = scaler.fit_transform(data[["Temperature"]])
    X_train, X_test = train_test_split(data, test_size=0.33)

    preprocess_and_save_data(X_train, train_output_folder)
    preprocess_and_save_data(X_test, test_output_folder)

if __name__ == "__main__":
    main()
