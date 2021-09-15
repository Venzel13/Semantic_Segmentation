from os import listdir

import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import *


class Cityscapes(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, sample):
        image_mask = Image.open(self.img_paths[sample])
        w, h = image_mask.size
        image = image_mask.crop((0, 0, w//2, h))
        mask = image_mask.crop((w//2, 0, w, h))
        image, mask = np.array(image), np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def get_path(dir_path, fold):
    rel_paths = np.array(listdir(dir_path + fold), dtype='object')
    abs_path = dir_path + fold + '/' + rel_paths
    return abs_path


transforms = {
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
    'test': None
        #TODO how to unnormalize?
}




datasets = {
    fold: Cityscapes(get_path(dir_path, fold), transforms[fold]) for fold in ["train", "val", "test"]
}

dataloaders = {
    'train' : DataLoader(datasets['train'], batch_size=64, shuffle=True, drop_last=True, pin_memory=True),
    'val' : DataLoader(datasets['val'], batch_size=256, shuffle=False),
    'test' : DataLoader(datasets['test'], batch_size=256, shuffle=False),
}