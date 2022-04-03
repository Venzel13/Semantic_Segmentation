import albumentations as A
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import EarlyStopping
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification.iou import IoU


# TODO hydra!
# TODO poetry instead of -r requirements.txt
DIR_PATH = "/home/eduard_kustov/data/"
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
LR = 1e-2
LOSS = DiceLoss(mode='multiclass') #TODO + FocalLoss(mode='multiclass')
METRIC = IoU(num_classes=N_CLASSES)
SCHEDULER = ReduceLROnPlateau
CALLBACKS = [EarlyStopping('val_loss', patience=5)]