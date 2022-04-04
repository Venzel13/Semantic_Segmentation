from os import listdir
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_img_mask_paths(dirname: str, fold: Literal['images', 'masks']) -> np.ndarray:
    assert fold in {'images', 'masks'}
    paths = dirname + fold + '/' + np.array(listdir(dirname + fold), dtype='object')
    return np.sort(paths)


def plot_random_pred_mask(pred: torch.Tensor, mask: torch.Tensor) -> None:
    i = np.random.choice(len(pred))
    plt.subplot(121)
    plt.title('true')
    plt.imshow(mask[i])
    plt.subplot(122)
    plt.title('pred')
    plt.imshow(pred[i])