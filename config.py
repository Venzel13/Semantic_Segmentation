import albumentations as A
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification.iou import IoU


# TODO hydra!
# TODO poetry instead of -r requirements.txt
DATA_PATH = "/home/eduard_kustov/data/"
CHECKPOINT_PATH = '/home/eduard_kustov/segmentation/lightning_logs/version_2/checkpoints/epoch=31-step=383.ckpt'
BATCH_SIZE = (37, 50, 55)
TEST_TRANSFORMS = A.Compose(
    [
        A.SmallestMaxSize(256),
        A.CenterCrop(256, 256),
        A.Normalize()
    ]
)
TRANSFORMS = {
    'train': A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.Rotate(limit=90),
            A.Normalize()
        ]),
    'val': TEST_TRANSFORMS,
    'test': TEST_TRANSFORMS
}
N_CLASSES = 2
N_CHANNELS = 3
ENCODER_NAME = 'resnet101'
MODEL = smp.DeepLabV3Plus(
    encoder_name=ENCODER_NAME,
    encoder_weights='imagenet',
    in_channels=N_CHANNELS,
    classes=N_CLASSES,
)
OPTIMIZER = Adam
LR = 1e-4
LOSS = DiceLoss(mode='multiclass') #TODO + FocalLoss(mode='multiclass')
METRIC = IoU(num_classes=N_CLASSES)
SCHEDULER = ReduceLROnPlateau
CALLBACKS = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        every_n_epochs=1,
    )
]