from os import listdir
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import *
from utils import *


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
    def __init__(self, img_paths: List[str], mask_paths: List[str], transforms: A.Compose = None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, sample) -> Tuple[np.ndarray, np.ndarray]:
        image = Image.open(self.img_paths[sample])
        mask = Image.open(self.mask_paths[sample])
        image, mask = np.array(image), np.array(mask)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


class LeafDataModule(pl.LightningDataModule):
    # def __init__(self, img_paths: str, mask_paths:str, transforms: Dict[str, A.Compose] = typical_transforms, bs: Tuple[int, int, int]):
    def __init__(self, dirpath: str, transforms: Dict[str, A.Compose] = typical_transforms, bs: Tuple[int, int, int] = (32, 20, 20)):
        super().__init__()
        self.dirpath = dirpath
        # self.img_paths = img_paths
        # self.mask_paths = mask_paths
        self.transforms = transforms
        self.bs = bs

    # def prepare_data(self) -> None:
    #     pass

    def setup(self) -> None:



        filenames = {}
        filenames['train'], filenames['test'] = train_test_split(listdir(self.img_path), test_size=0.1)
        filenames['train'], filenames['val'] = train_test_split(filenames['train'], test_size=0.1)

        self.datasets = {
            fold: Leaf(
                img_paths=get_img_mask_paths(self.dirpath, 'images'),
                mask_paths=get_img_mask_paths(self.dirpath, 'masks'),
                transforms=self.transforms[fold]
     
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



a = LeafDataModule()



# sizes_1 = []
# sizes_2 = []
# for img_path in listdir('C:/Users/Eduard_Kustov/Desktop/learn/CV/segmentation/data/masks/'):
#     image = Image.open('C:/Users/Eduard_Kustov/Desktop/learn/CV/segmentation/data/masks/' + img_path)
#     sizes_2.append([img_path, image.size[0], image.size[1]])


# df = pd.DataFrame(sizes_1)
# df2 = pd.DataFrame(sizes_2)

# df.shape
# df[(df[1] > 255) | (df[2] > 255)]

