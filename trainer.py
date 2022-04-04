import pytorch_lightning as pl

from model import LeafModule
from preproc import LeafDataModule
from config import DATA_PATH, CALLBACKS


model = LeafModule()
data = LeafDataModule(DATA_PATH)
trainer = pl.Trainer(
    gpus=1,
    callbacks = CALLBACKS,
    log_every_n_steps=12,
)
trainer.fit(model, data)
#TODO вынести параметры pl.Trainer в config
#TODO from argparse import ArgumentParser