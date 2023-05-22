import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config


def fit_epoch(
        model: nn.Sequential,
        train_loader: Dataset,
        criterion: nn.Sequential,
        optimizer: torch.optim
) -> tuple['float', 'float']:
    """
    Обучение одной эпохи
    """

    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_data += inputs.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.to(config.device_cpu).numpy() / processed_data
    
    return train_loss, train_acc
