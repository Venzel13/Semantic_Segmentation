from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import *
from utils import *


typical_transforms = {
    'train': A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30),
            A.Normalize() # уже с imagenet по умолчанию
        ]),
    'val': A.Compose(
        [
            A.Normalize()
        ]),
    'test': A.Compose(
        [
           A.Normalize() #TODO how to unnormalize?
        ]
    )
}


class Cityscapes(Dataset):
    def __init__(self, img_paths: List[str], transforms: A.Compose = None):
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, sample) -> Tuple[np.ndarray, np.ndarray]:
        image_mask = Image.open(self.img_paths[sample])
        w, h = image_mask.size
        image = image_mask.crop((0, 0, w//2, h))
        mask = image_mask.crop((w//2, 0, w, h))
        image, mask = np.array(image), np.array(mask)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


class CityscapesDataModule(pl.LightningDataModule):

    def __init__(self, dir_path: str, transforms: Dict[str, A.Compose] = typical_transforms, bs_train: int = 64, bs_val: int = 250, bs_test: int = 250):
        super().__init__()
        self.dir_path = dir_path
        self.transforms = transforms
        self.bs_train = bs_train
        self.bs_val = bs_val
        self.bs_test = bs_test

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str]) -> None:
        self.datasets = {
            fold: Cityscapes(
                img_paths=get_path(self.dir_path, fold),
                transforms=self.transforms[fold]
            )
            for fold in ["train", "val", "test"]
        }

    def train_dataloader(self) -> DataLoader:
        train = DataLoader(
            self.datasets['train'],
            batch_size=self.bs_train,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        return train

    def val_dataloader(self) -> DataLoader:
        val = DataLoader(self.datasets['val'], batch_size=self.bs_val, shuffle=False)
        return val

    def test_dataloader(self) -> DataLoader:
        test = DataLoader(self.datasets['test'], batch_size=self.bs_test, shuffle=False)
        return test


# a = CityscapesDataModule(dir_path, transforms)
# a.setup('test')
# next(iter(a.test_dataloader()))
