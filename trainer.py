import pytorch_lightning as pl

from model import LeafModule
from preproc import LeafDataModule
from config import DIR_PATH, N_CLASSES

model = LeafModule(n_classes=N_CLASSES)
data = LeafDataModule(DIR_PATH)
trainer = pl.Trainer()
trainer.fit(model, data)

#####
# from torchmetrics.classification.iou import IoU
# from torchmetrics.classification.f_beta import F1
# data.setup('predict')
# val = data.val_dataloader()
# image, mask = next(iter(val))
# pred = model(image)
# pred = pred.log_softmax(dim=1).exp()