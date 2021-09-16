from os import listdir
from typing import List #, Literal

import numpy as np


def get_path(dir_path: str, fold: str) -> List[str]:
    rel_paths = np.array(listdir(dir_path + fold), dtype='object')
    abs_path = dir_path + fold + '/' + rel_paths
    return abs_path
