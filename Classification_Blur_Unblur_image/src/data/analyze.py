import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import config

matplotlib.use('Agg')


def report_value_counts(data_labels: pd.DataFrame) -> None:
    """
    Возвращает количественное распределение признаков
    """
    sns.catplot(data=data_labels, x="blur", kind="count")
    plt.savefig('./Report/value_counts_blure.png')
    plt.switch_backend('agg')


def labels(directory: str) -> pd.DataFrame:
    """
    Считываем таргет из файла
    """
    data_labels = pd.read_csv(directory + config.train_file)
    data_labels[['blur']] = data_labels[['blur']].astype('int')

    return data_labels
