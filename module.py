import pytorch_lightning as pl
import torch


class LeafModule(pl.LightningModule):
    def __init__(self, model, loss, metric, optimizer, scheduler, monitor):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor

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

    def predict_step(self, predict_batch: torch.Tensor,
                     batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        images, masks = predict_batch
        pred = self(images).argmax(1)
        return pred, masks

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": self.monitor
        }

    def step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images, masks = batch
        masks = masks.long()
        logits = self(images)
        loss = self.loss(logits, masks)
        pred_class = logits.argmax(1)
        metric = self.metric(pred_class, masks)
        
        return loss, metric