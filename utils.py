from os import listdir
from typing import List, Literal

import numpy as np


def get_img_mask_paths(dirname: str, fold: Literal['images', 'masks']) -> List[str]:
    assert fold in {'images', 'masks'}
    return dirname + fold + '/' + np.array(listdir(dirname + fold), dtype='object')