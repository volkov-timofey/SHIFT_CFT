from tqdm import tqdm
import torch
from typing import Callable
import torch.nn as nn
from typing import Tuple

def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable
) -> Tuple[float, float]:

    model.eval()

    val_running_loss = 0.0
    val_running_correct = 0

    with torch.no_grad():
        inference_steps = int(len(dataloader.dataset)/dataloader.batch_size)

        for _, batch in tqdm(enumerate(dataloader), total=inference_steps):
            
            images, target = batch
            images = images.to(device)
            target = target.to(device)

            outputs = model(images)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/len(dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(dataloader.dataset)        
        return val_loss, val_accuracy