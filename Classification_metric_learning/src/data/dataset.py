from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CastomDataset(Dataset):

    def __init__(self, df: pd.DataFrame, mode, transforms=None):
        self.df = df[df.split == mode]
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index, :]
        # принудительно конвертируем в RGB из RGBA
        image = Image.open(row['image']).convert('RGB')

        label = row['int_label']

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.df)
