from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import config
from src.data.create_df import pivot_df_data
from src.data.dataset import CastomDataset
from src.data.download_data import data_extract
from src.modeling.fix_seed import seed_everything
from src.modeling.model_simpleNN import MLP
from src.modeling.train import train

wandb.login()



def pipeline() -> None:
    """
    Проходим полный путь обучения
    """


    # Инициализируем wandb
    wandb.init(project='metric-learning')

    # fix random
    seed_everything(config.seed)

    # Извлекаем файлы 1 раз, пропускаем при их наличии
    if Path(config.main_dir + config.data_dir + config.train_dir).is_dir() == False:
        data_extract(
            config.main_dir + config.data_dir + config.zip_file,
            config.main_dir + config.data_dir
        )

    # Извлекаем лейблы...формируем даталоадер
    df = pivot_df_data(config.main_dir + config.data_dir)

    train_dataset = CastomDataset(
        df,
        mode='train',
        transforms=config.train_transform
    )

    val_dataset = CastomDataset(
        df,
        mode='validation',
        transforms=config.val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    trunk = config.model
    trunk_output_size = trunk.fc.in_features
    trunk.fc = nn.Identity()
    trunk = nn.DataParallel(trunk.to(config.device))

    embedder = nn.DataParallel(MLP([trunk_output_size, config.embeddings]).to(config.device))

    classifier = nn.DataParallel(MLP([config.embeddings, config.n_classes])).to(config.device)

    models = {"trunk": trunk, "embedder": embedder, "classifier": classifier}


    # обучение модели

    train(train_loader,
          val_loader,
          models,
          config.epochs)
