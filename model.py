import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchmetrics  # TODO будет в __init__() lightning
# from segmentation_models_pytorch.utils.losses import JaccardLoss
from segmentation_models_pytorch.losses import JaccardLoss
from torch.optim import Adam

#TODO сделать через Hydra или Gin-config (подавать строки, а не классы в params)
#TODO kwargs

model = smp.DeepLabV3Plus(
    encoder_name='resnet101',
    encoder_weights='imagenet',
    # activation='softmax', #TODO нам нужны логиты
    in_channels=3,
    classes=30,
)


class LeafModule(pl.LightningModule):
    def __init__(self, model=model, optimizer=Adam, lr=1e-3, loss=JaccardLoss, metric=None, n_classes=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.metric = metric
        self.n_classes = n_classes
        # pass

    def forward(self): #TODO нужен ли?
        pass

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters, self.lr)
        return optimizer

    def training_step(self, train_batch): #TODO add batch_idx?!?!
        images, masks = train_batch
        logits = self.model(images)
        loss = self.loss()

        pass
        # return loss

    def validation_step(self, val_batch, batch_idx):
        pass
        # return loss


import torch
from segmentation_models_pytorch.losses import JaccardLoss
pred = torch.rand(10, 256, 256, dtype=torch.float32)
true = torch.rand(10, 256, 256, dtype=torch.float32)
loss = JaccardLoss(mode='multiclass')
loss(pred, true)


loss = JaccardLoss(mode='binary')
output = model(torch.rand(10, 3, 256, 256))




loss(output, true)
output.size()

output.log_softmax(dim=1).exp().size()

true.view(10, -1).size()
output.view(10, 30, ).size()