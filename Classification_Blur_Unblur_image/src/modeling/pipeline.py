from pathlib import Path

import wandb
import yaml
from torch.utils.data import DataLoader

from src.data.analyze import labels, report_value_counts
from src.data.dataset import CastomDataset
from src.data.download_data import data_extract
from src.data.split import data_split
from src.modeling.fix_seed import seed_everything
from src.modeling.predict import roc_auc_round
from src.modeling.train import train

wandb.login()

config_yaml = dict()
with open('config_experiments.yaml') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)


def pipeline(config=None) -> None:
    """
    Проходим полный путь обучения,
    завернутый в wandb
    """
    import config

    # Инициализируем wandb
    wandb.init()
    config_wandb = wandb.config

    # извлекаем параметры для обучения
    lr = config_wandb.learning_rate
    optimizer = config_wandb.optimizer
    gamma = config_wandb.gamma
    bs = config_wandb.batch_size

    # fix random
    seed_everything(config.seed)

    # Извлекаем файлы 1 раз, пропускаем при их наличии
    if Path(config.main_dir + config.train_dir).is_dir() == False:
        data_extract(config.zip_file, config.data_dir)

    # Извлекаем лейблы...формируем даталоадер
    data_labels = labels(config.data_dir)
    report_value_counts(data_labels)
    train_files, val_files, test_files = data_split(data_labels)

    train_dataset = CastomDataset(train_files, mode='train')
    val_dataset = CastomDataset(val_files, mode='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=bs,
                              shuffle=True,
                              drop_last=True,
                              num_workers=2,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=bs,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    # обучение модели

    train(
        train_loader,
        val_loader,
        config.model,
        config.epochs,
        lr,
        optimizer,
        gamma)

    roc_auc = roc_auc_round(val_dataset)

    # логируем roc_auc
    wandb.log({"roc_auc": roc_auc})


sweep_id = wandb.sweep(sweep=config_yaml, project='requrements_hyperparameters')
wandb.agent(sweep_id, function=pipeline, count=10)
