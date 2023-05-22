from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def data_split(data_labels: pd.DataFrame) -> tuple[List[str], List[str], List[str]]:
    """
    создает датасеты для обучения, валидации и теста
    """
    train_dir = Path(config.data_dir + config.train_dir)
    test_dir = Path(config.data_dir + config.test_dir)

    train_val_files = list(train_dir.rglob('*.jpg'))
    test_files = list(test_dir.rglob('*.jpg'))

    train_val_labels = [np.array(data_labels[data_labels.iloc[:, 0] == path.name].iloc[:, 1])[0] for path in
                        train_val_files]
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=config.test_size,
        stratify=train_val_labels,
        random_state=config.seed
    )

    return train_files, val_files, test_files
