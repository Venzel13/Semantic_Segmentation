import pytorch_lightning as pl

from model import LeafModule
from preproc import LeafDataModule
from config import *

model = LeafModule(n_classes=N_CLASSES)
data = LeafDataModule(dir_path)
trainer = pl.Trainer()
trainer.fit(model, data)