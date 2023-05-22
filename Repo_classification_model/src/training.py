from tqdm import tqdm
import torch
from typing import Tuple, Callable
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
) -> Tuple[float, float]:

    model.train() 

    train_running_loss = 0.0
    train_running_correct = 0
    total_epoch_steps = int(len(dataloader.dataset)/dataloader.batch_size)

    for _, batch in tqdm(enumerate(dataloader), total=total_epoch_steps):
        images, target = batch
        images, target = images.to(device), target.to(device)
        outputs = model(images)
        
        loss = criterion(outputs, target)
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1) # можем не использовать softmax, а просто взять .max
        train_running_correct += (preds == target).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/len(dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(dataloader.dataset)    
    return train_loss, train_accuracy