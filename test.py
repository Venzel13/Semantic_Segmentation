import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import CALLBACKS, DATA_PATH
from model import LeafModule
from preproc import LeafDataModule

log = '/home/eduard_kustov/segmentation/lightning_logs/version_2/checkpoints/epoch=31-step=383.ckpt'
checkpoint_callback = ModelCheckpoint(log, monitor='val_loss', save_top_k=1)

model = LeafModule.load_from_checkpoint(log)
data = LeafDataModule(DATA_PATH)
trainer = pl.Trainer(callbacks=CALLBACKS)

trainer.checkpoint_callback.best_model_path # empty string
