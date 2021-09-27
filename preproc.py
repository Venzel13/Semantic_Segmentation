from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import get_img_mask_paths


typical_transforms = {
    'train': A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.Rotate(limit=90),
            A.Normalize()
        ]),
    'val': A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(256, 256),
            A.Normalize()
        ]),
    'test': A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(256, 256),
            A.Normalize()
        ]
    )
}


class Leaf(Dataset):
    def __init__(self, img_paths: List[str], mask_paths: List[str], transform: A.Compose = None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, sample) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.img_paths[sample])
        mask = cv2.imread(self.mask_paths[sample], cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
        mask = mask.astype('int64')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


class LeafDataModule(pl.LightningDataModule):
    def __init__(self, dirpath: str, transforms: Dict[str, A.Compose] = typical_transforms, bs: Tuple[int, int, int] = (32, 50, 55)):
        super().__init__()
        self.dirpath = dirpath
        self.transforms = transforms
        self.bs = bs

    def setup(self, stage) -> None:
        img_paths = {}
        mask_paths = {}

        img_paths['train'], img_paths['test'], mask_paths['train'], mask_paths['test'] = train_test_split(
            get_img_mask_paths(self.dirpath, 'images'),
            get_img_mask_paths(self.dirpath, 'masks'),
            test_size=0.1,
            random_state=17
        )
        img_paths['train'], img_paths['val'], mask_paths['train'], mask_paths['val'] = train_test_split(
            img_paths['train'],
            mask_paths['train'],
            test_size=0.1,
            random_state=17,
        )
        
        self.datasets = {
            fold: Leaf(
                img_paths=img_paths[fold],
                mask_paths=mask_paths[fold],
                transform=self.transforms[fold]
            )
            for fold in ["train", "val", "test"]
        }

    def train_dataloader(self) -> DataLoader:
        train = DataLoader(
            self.datasets['train'],
            batch_size=self.bs[0],
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        return train

    def val_dataloader(self) -> DataLoader:
        val = DataLoader(self.datasets['val'], batch_size=self.bs[1], shuffle=False)
        return val

    def test_dataloader(self) -> DataLoader:
        test = DataLoader(self.datasets['test'], batch_size=self.bs[2], shuffle=False)
        return test