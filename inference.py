from typing import Union

import hydra
import torch
from omegaconf import DictConfig

from utils import instantiate_objects


@hydra.main(config_path='conf', config_name='config')
def make_inference(cfg: DictConfig) \
                   -> Union[tuple[torch.Tensor, torch.Tensor], dict[str, float]]:
    assert cfg.inference.stage in {'test', 'predict'}

    module, datamodule, trainer = instantiate_objects(cfg)
    module.load_from_checkpoint(
        cfg.inference.best_model_path,
        model=module.model,
        loss=module.loss,
        metric=module.metric,
        optimizer=module.optimizer,
        scheduler=module.scheduler,
        monitor=module.monitor
    )
    if cfg.inference.stage == 'test':
        score = trainer.test(module, datamodule)
        return score[0]

    pred = trainer.predict(module, datamodule)
    pred, mask = pred[0]
    return pred, mask

make_inference()