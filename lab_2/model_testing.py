import os
import glob
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score


def load_test_data(train_folder):
    df = pd.read_csv(os.path.join(train_folder, 'dataset.csv'))
    if "Day" not in df.columns or "Temperature" not in df.columns:
        raise ValueError(f"Файл dataset.csv не содержит необходимых колонок 'Day' и 'Temperature'.")
    return df


def main():
    test_folder = "test_preprocessed"
    data = load_test_data(test_folder)

    X_test = data[["Day"]].values
    y_test = data["Temperature"].values

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Результаты тестирования модели на данных из папки 'test':")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()
