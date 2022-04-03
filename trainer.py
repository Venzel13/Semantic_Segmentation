import pytorch_lightning as pl

from model import LeafModule
from preproc import LeafDataModule
from config import DIR_PATH, CALLBACKS


model = LeafModule()
data = LeafDataModule(DIR_PATH)
trainer = pl.Trainer(
    # gpus=1,
    callbacks = CALLBACKS,
    detect_anomaly=True,
    log_every_n_steps = 1,
)
trainer.fit(model, data)

#TODO trainer.test(model, data) #automatically load best_model #or choose best_path!! ckpt_path='best'
#TODO from argparse import ArgumentParser