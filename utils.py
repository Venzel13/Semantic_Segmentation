from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig


def get_img_mask_paths(dirname: str, fold: Literal['images', 'masks']) -> list[str]:
    assert fold in {'images', 'masks'}
    paths = sorted(Path(dirname).joinpath(fold).iterdir())
    paths = [str(path) for path in paths]
    return paths


def instantiate_objects(cfg: DictConfig) -> \
                        tuple[pl.LightningModule, pl.LightningDataModule, pl.Trainer]:
    model = instantiate(cfg.module.model)
    optimizer = instantiate(cfg.module.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.module.scheduler, optimizer=optimizer)
    module = instantiate(cfg.module, model=model, optimizer=optimizer, scheduler=scheduler)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)

    return module, datamodule, trainer


def plot_random_pred_mask(pred: torch.Tensor, mask: torch.Tensor) -> None:
    i = np.random.choice(len(pred))
    plt.subplot(121)
    plt.title('true')
    plt.imshow(mask[i])
    plt.subplot(122)
    plt.title('pred')
    plt.imshow(pred[i])