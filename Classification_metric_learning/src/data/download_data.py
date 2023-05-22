import os
import shutil
from pathlib import Path

import wandb

import config


def data_extract(filename: str, extract_dir: str) -> None:
    """
    Скачивание датасета
    Распаковка архива в необходимую директорию

    filename - путь до датасета /.zip
    extract_dir - директория для извлечения
    """

    if not Path(extract_dir).is_dir():
        os.makedirs(extract_dir)

        run = wandb.init()
        artifact = run.use_artifact(config.path_wandb, type='dataset')
        artifact.download(extract_dir)

        shutil.unpack_archive(filename, extract_dir)
        os.remove(filename)