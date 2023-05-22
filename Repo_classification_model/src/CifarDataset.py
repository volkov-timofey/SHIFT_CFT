from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple
from torchvision.transforms import transforms
import torch
import pandas as pd

class CifarDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, mode, transforms=None):
        self.df = df[df.split == mode]
        self.transforms = transforms
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index, :]
        image = Image.open(row['image'])
        label = row['int_label']

        if self.transforms:
            # Аугментация
            image = self.transforms(image)

        return image, label
    
    def __len__(self):
        return len(self.df)