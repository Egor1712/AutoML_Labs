import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

def load_training_data(train_folder):
    """
    Загружает данные из всех CSV файлов в указанной папке.
    Каждый файл должен содержать колонки "Day" и "Temperature".
    """
    csv_files = glob.glob(os.path.join(train_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"В папке {train_folder} не найдено CSV файлов.")
    
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        if "Day" not in df.columns or "Temperature" not in df.columns:
            raise ValueError(f"Файл {file} не содержит необходимых колонок 'Day' и 'Temperature'.")
        df_list.append(df)
    
    # Объединяем данные из всех файлов в один DataFrame
    return pd.concat(df_list, ignore_index=True)

def main():
    train_folder = "train_preprocessed"
    data = load_training_data(train_folder)
    
    # Определяем признаки (в данном случае только "Day") и целевую переменную ("Temperature")
    X = data[["Day"]].values
    y = data["Temperature"].values

    # Создаем и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(X, y)
    print("Модель обучена на данных из папки 'train_preprocessed'.")

    # Сохраняем модель в файл model.pkl
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Модель сохранена в файле model.pkl.")

if __name__ == "__main__":
    main()
