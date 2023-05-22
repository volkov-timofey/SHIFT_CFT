import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.modeling.eval_one_epoch import eval_epoch
from src.modeling.fit_one_epoch import fit_epoch
from src.modeling.save_training import save_training


def train(
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Sequential,
        epochs: int,
        lr: float,
        optimizer: str,
        gamma: float
) -> list[tuple[float, float, float, float]]:
    """
    Обучение модели, логирование гиперпараметров,
    сохранение наилучшей модели
    """

    def build_optimizer(optimizer: str, lr: float) -> torch.optim:
        if optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return optimizer

    optimizer_ = build_optimizer(optimizer, lr)

    history = []  # сохраняем данные о loss и accuracy для train и val

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} \
    current_lr {current_lr}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_, gamma=config.gamma)

        criterion = config.criterion

        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model,
                                              train_loader,
                                              criterion,
                                              optimizer_)

            print("loss", train_loss)

            val_loss, val_acc = eval_epoch(model, val_loader, criterion)

            scheduler.step()

            history.append((train_loss, train_acc, val_loss, val_acc))

            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "epoch": epoch,
                       "train_acc": train_acc,
                       "val_acc": val_acc,
                       "scheduler": optimizer_.param_groups[0]['lr'],
                       "optimizer": optimizer,
                       "gamma": gamma
                       })

            pbar_outer.update(1)
            tqdm.write(log_template.format(
                ep=epoch + 1,
                t_loss=train_loss,
                v_loss=val_loss,
                t_acc=train_acc,
                v_acc=val_acc,
                current_lr=optimizer_.param_groups[0]['lr']))

            save_training(
                epoch,
                model,
                optimizer_,
                name_model=f'model_in_{epoch}_epoch'
            )

            # сохраняем модель в логи wandb
            model.save(os.path.join(wandb.run.dir, f'model_in_{epoch}_epoch.h5'))

    return history
