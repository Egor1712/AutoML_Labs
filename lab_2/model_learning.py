import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


def load_training_data(train_folder):
    df = pd.read_csv(os.path.join(train_folder, 'dataset.csv'))
    if "Day" not in df.columns or "Temperature" not in df.columns:
        raise ValueError(f"Файл dataset.csv не содержит необходимых колонок 'Day' и 'Temperature'.")
    return df


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
