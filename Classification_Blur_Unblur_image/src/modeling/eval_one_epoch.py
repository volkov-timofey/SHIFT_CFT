import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config


def eval_epoch(model: nn.Sequential, val_loader: Dataset, criterion: nn.Sequential) -> tuple['float', 'float']:
    """
    Валидация одной эпохи
    """

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels.type(torch.LongTensor))
            preds = torch.argmax(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.double() / processed_size

    return val_loss, val_acc
