from typing import Dict, Literal, Tuple, Union

import pytorch_lightning as pl
import torch

from model import LeafModule
from preproc import LeafDataModule


def make_inference(data_path:str, checkpoint_path:str,
                   stage: Literal['test', 'predict'] = 'predict') \
                   -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, float]]:
    assert stage in {'test', 'predict'}
    model = LeafModule.load_from_checkpoint(checkpoint_path)
    data = LeafDataModule(data_path)
    trainer = pl.Trainer(enable_checkpointing=False)
    
    if stage == 'test':
        score = trainer.test(model, data)
        return score[0]

    pred = trainer.predict(model, data)
    pred, mask = pred[0]
    return pred, mask