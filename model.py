from typing import Tuple

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.losses import JaccardLoss
from torch.optim import Adam
from torchmetrics.classification.iou import IoU

from config import N_CLASSES

model = smp.DeepLabV3Plus(
    encoder_name='resnet101',
    encoder_weights='imagenet',
    in_channels=3,
    classes=N_CLASSES,
)

class LeafModule(pl.LightningModule):
    def __init__(self, model=model, n_classes=None, optimizer=Adam, lr=1e-3, loss=JaccardLoss(mode='multiclass'), metric=IoU):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.metric = metric
        self.n_classes = n_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.permute(0, 3, 1, 2)
        logits = self.model(images)
        return logits

    def training_step(self, train_batch: torch.Tensor, batch_idx) -> torch.Tensor:
        train_loss, train_metric = self.step(train_batch)
        self.log('train', {'loss': train_loss, 'metric': train_metric})
        return train_loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx) -> torch.Tensor:
        val_loss, val_metric = self.step(val_batch)
        self.log('val', {'loss': val_loss, 'metric': val_metric})
        return val_loss

    def test_step(self, test_batch: torch.Tensor, batch_idx) -> torch.Tensor:
        test_loss, test_metric = self.step(test_batch)
        self.log('test', {'loss': test_loss, 'metric': test_metric})
        return test_loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images, masks = batch
        logits = self(images)
        loss = self.loss(logits, masks)
        metric = self.metric(self.n_classes) #too much time
        metric = metric(logits, masks)
        return loss, metric