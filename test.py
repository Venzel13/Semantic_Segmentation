import pytorch_lightning as pl

from config import DIR_PATH
from model import LeafModule
from preproc import LeafDataModule

#TODO trainer.test(model, data, ckpt_path='best')
#TODO trainer.predict() ?!

model = LeafModule.load_from_checkpoint('/lightning_logs/version_1/checkpoints/epoch=49-step=599.ckpt')
data = LeafDataModule(DIR_PATH)


trainer = pl.Trainer()
trainer.test(model, data, ckpt_path='best')