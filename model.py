from typing import Tuple

import pytorch_lightning as pl
import torch
from config import MODEL, OPTIMIZER, LR, LOSS, METRIC


class LeafModule(pl.LightningModule):
    def __init__(self, model=MODEL, optimizer=OPTIMIZER, lr=LR, loss=LOSS, metric=METRIC):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.metric = metric

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.permute(0, 3, 1, 2)
        logits = self.model(images)
        return logits

    def training_step(self, train_batch: torch.Tensor, batch_idx) -> None:
        loss, metric = self.step(train_batch)
        self.log('train_loss', loss)
        self.log('train_metric', metric)

    def validation_step(self, val_batch: torch.Tensor, batch_idx) -> None:
        loss, metric = self.step(val_batch)
        self.log('val_loss', loss)
        self.log('val_metric', metric)

    def test_step(self, test_batch: torch.Tensor, batch_idx) -> None:
        loss, metric = self.step(test_batch)
        self.log('test_loss', loss)
        self.log('test_metric', metric)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        #TODO LR scheduler ReduceLROnPlateau
        return optimizer

    def step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images, masks = batch
        masks = masks.long()
        logits = self(images)
        loss = self.loss(logits, masks)
        pred_class = logits.argmax(1)
        metric = self.metric(pred_class, masks)
        
        return loss, metric