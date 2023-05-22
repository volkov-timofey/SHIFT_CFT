import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import config
from src.data.analyze import labels


class CastomDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, files: np.array('str'), mode: list[str]):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode

        if self.mode not in config.data_modes:
            print(f"{self.mode} is not correct; correct modes: {config.data_modes}")
            raise NameError

        self.len_ = len(self.files)

        # загружем метки файлов
        if self.mode != 'test':
            self.labels = [
                np.array(labels(config.data_dir)[labels(config.data_dir).iloc[:, 0] == path.name].iloc[:, 1])[0] \
                for path in self.files
            ]

    def __len__(self):
        return self.len_

    def load_sample(self, file: np.array('str')) -> Image:
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        # проводим дополнительную обработку изображений
        # переводим в тензоры и нормализуем
        global transform
        if self.mode == 'train':
            transform = config.transform_train

        if (self.mode == 'val') or (self.mode == 'test'):
            transform = config.transform_test

        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            y = self.labels[index]
            return x, y

    def _prepare_sample(self, image) -> np.array('image'):
        image = image.resize((config.rescale_size, config.rescale_size))
        return np.array(image)
