import pandas as pd
import os
from catboost.datasets import titanic
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Загружаем данные
    train, _ = titanic()
    df = train[['Pclass', 'Sex', 'Age']]
    df.to_csv(os.path.join(cfg.output_dir, cfg.raw_file), index=False)

    # Заполняем пропуски
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df.to_csv(os.path.join(cfg.output_dir, cfg.filled_file), index=False)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df.to_csv(os.path.join(cfg.output_dir, cfg.encoded_file), index=False)

    # Конечный файл
    df.to_csv(os.path.join(cfg.output_dir, cfg.processed_file), index=False)

    print("Data processing complete.")

if __name__ == "__main__":
    main()