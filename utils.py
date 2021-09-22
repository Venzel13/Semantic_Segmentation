from os import listdir
from typing import List, Literal

import numpy as np


def get_img_mask_paths(dirname: str, fold: Literal['images', 'masks']) -> List[str]:
    folds = {'images', 'masks'}
    if fold not in folds:
        raise ValueError("fold must be one of {}".format(folds))
    return dirname + fold + '/' + np.array(listdir(dirname + fold), dtype='object')