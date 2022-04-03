from typing import Tuple

import pytorch_lightning as pl
import torch
from config import MODEL, OPTIMIZER, LR, LOSS, METRIC, SCHEDULER


class LeafModule(pl.LightningModule):
    def __init__(self, model=MODEL, lr=LR, loss=LOSS, metric=METRIC,
                 optimizer=OPTIMIZER, scheduler=SCHEDULER):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.permute(0, 3, 1, 2)
        logits = self.model(images)
        return logits

    def training_step(self, train_batch: torch.Tensor, batch_idx) -> torch.Tensor:
        loss, metric = self.step(train_batch)
        self.log('train_loss', loss)
        self.log('train_metric', metric)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx) -> None:
        loss, metric = self.step(val_batch)
        self.log('val_loss', loss)
        self.log('val_metric', metric)

    def test_step(self, test_batch: torch.Tensor, batch_idx) -> None:
        loss, metric = self.step(test_batch)
        self.log('test_loss', loss)
        self.log('test_metric', metric)

    # def predict_step(self, predict_batch: torch.Tensor, batch_idx) -> torch.Tensor:
    #     images, _ = predict_batch
    #     pred = self(images)
    #     return pred

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        scheduler = self.scheduler(optimizer) #TODO config with params for each param **kwargs
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images, masks = batch
        masks = masks.long()
        logits = self(images)
        loss = self.loss(logits, masks)
        pred_class = logits.argmax(1)
        metric = self.metric(pred_class, masks)
        
        return loss, metric