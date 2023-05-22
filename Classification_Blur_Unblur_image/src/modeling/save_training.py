import torch
import torch.nn as nn

import config


def save_training(
        current_epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        name_model: str
) -> None:
    """
    сохраняем текущее состояние модели
    """
    save_path = config.save_path_ + f'{name_model}.ckpt'

    torch.save({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
