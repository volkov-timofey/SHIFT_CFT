from torchvision.transforms import transforms
import torch
from src.CifarDataset import CifarDataset
import pandas as pd
from typing import Tuple
import config


def CifarDataLoader(
    df: pd.DataFrame
  ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2023, 0.1994, 0.2010)),
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CifarDataset(
        df, 
        mode='train',
        transforms=transform_train
    )

    test_dataset = CifarDataset(
        df, 
        mode='test', 
        transforms=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader