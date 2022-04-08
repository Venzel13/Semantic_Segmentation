from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import get_img_mask_paths


class Leaf(Dataset):
    def __init__(self, img_paths: np.ndarray, mask_paths: np.ndarray,
                 transform: Optional[A.Compose] = None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, sample) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.img_paths[sample])
        mask = cv2.imread(self.mask_paths[sample], cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


class LeafDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, transforms: dict[str, A.Compose],
                 batch_size: dict[str, int], n_workers: int):
        super().__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_workers = n_workers

    def setup(self, stage) -> None:
        img_paths = {}
        mask_paths = {}

        (img_paths['train'], img_paths['test'],mask_paths['train'],
         mask_paths['test']) = train_test_split(
            get_img_mask_paths(self.data_path, 'images'),
            get_img_mask_paths(self.data_path, 'masks'),
            test_size=0.1,
            random_state=17
        )
        (img_paths['train'], img_paths['val'], mask_paths['train'],
         mask_paths['val']) = train_test_split(
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
            batch_size=self.batch_size['train'],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.n_workers
        )
        return train

    def val_dataloader(self) -> DataLoader:
        val = DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size['val'],
            shuffle=False,
            num_workers=self.n_workers
        )
        return val

    def test_dataloader(self) -> DataLoader:
        test = DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size['test'],
            shuffle=False,
            num_workers=self.n_workers
        )
        return test

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()