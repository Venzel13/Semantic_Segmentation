import hydra
from omegaconf import DictConfig

from utils import instantiate_objects


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig) -> None:
    module, datamodule, trainer = instantiate_objects(cfg)
    trainer.fit(module, datamodule)

train()