import os
import glob
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def load_test_data(test_folder):
    """
    Загружает данные из всех CSV файлов в указанной папке.
    Каждый файл должен содержать колонки "Day" и "Temperature".
    """
    csv_files = glob.glob(os.path.join(test_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"В папке {test_folder} не найдено CSV файлов.")
    
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        if "Day" not in df.columns or "Temperature" not in df.columns:
            raise ValueError(f"Файл {file} не содержит необходимых колонок 'Day' и 'Temperature'.")
        df_list.append(df)
    
    # Объединяем данные из всех файлов в один DataFrame
    return pd.concat(df_list, ignore_index=True)

def main():
    test_folder = "test_preprocessed"
    data = load_test_data(test_folder)
    
    # Определяем признаки и целевую переменную для тестовой выборки
    X_test = data[["Day"]].values
    y_test = data["Temperature"].values

    # Загружаем обученную модель из файла
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Выполняем предсказание
    y_pred = model.predict(X_test)
    
    # Оцениваем качество модели с использованием MSE и R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Результаты тестирования модели на данных из папки 'test':")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()
