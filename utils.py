from os import listdir
from typing import Literal

import numpy as np


def get_img_mask_paths(dirname: str, fold: Literal['images', 'masks']) -> np.ndarray:
    assert fold in {'images', 'masks'}
    paths = dirname + fold + '/' + np.array(listdir(dirname + fold), dtype='object')
    return np.sort(paths)