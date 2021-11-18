import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
from torch.optim import Adam
from torchmetrics.classification.iou import IoU

DIR_PATH = 'C:/Users/Eduard_Kustov/Desktop/learn/CV/segmentation/data/'

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
    'val': A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(256, 256),
            A.Normalize()
        ]),
    'test': A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(256, 256),
            A.Normalize()
        ]
    )
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
LR = 1e-3
LOSS = JaccardLoss(mode='multiclass')
METRIC = IoU

# TODO gin-config (gin.register, gin external configurable)
# TODO добавить callbacks early stopping, lr_scheduler (или его в configure_optimizer)
