import matplotlib.pyplot as plt
from src.CifarDataLoader import CifarDataLoader
from src.training import train_one_epoch
from src.validating import validate
from src.save_training import save_training
import torch
from typing import Callable
import pandas as pd
import torch.nn as nn

def train_model(
    model: nn.Module, 
    learning_rate: float, 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable, 
    epochs: int, 
    df: pd.DataFrame, 
    current_epoch: int = 0):

    train_loader, test_loader = CifarDataLoader(df)

    train_loss , train_accuracy = [], []

    for epoch in range(current_epoch, epochs):

        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = train_one_epoch(
                                                          model,
                                                          train_loader, 
                                                          optimizer, 
                                                          criterion
                                                        )

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)

        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")

        val_loss, val_accuracy = validate(model, test_loader, criterion)
        print(f"Test Loss: {val_loss:.4f}, Test Acc: {val_accuracy:.2f}")

        if epoch == 15:
          save_training(
                    epoch, 
                    model, 
                    optimizer,
                    name_model=f'model_{epoch}_epochs'
          )

    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path_ + '/Report/train_acc_epoch.png')
    
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_ + '/Report/train_loss_epoch.png')

    save_training(
                    epoch, 
                    model, 
                    optimizer,
                    name_model=f'final_model'
          )