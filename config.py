import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss
from torch.optim import Adam
from torchmetrics.classification.iou import IoU
from torch.nn import CrossEntropyLoss


# TODO gin-config (gin.register, gin external configurable)
DIR_PATH = '/home/eduard_kustov/one_batch/'
BATCH_SIZE = (23, 23, 23)
TEST_TRANSFORMS = A.Compose(
    [
        A.SmallestMaxSize(256),
        A.CenterCrop(256, 256),
        A.Normalize()
    ]
)
TRANSFORMS = {
    'train': TEST_TRANSFORMS,
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
LR = 1e-5
LOSS = DiceLoss(mode='multiclass') #TODO add focal loss
METRIC = IoU(num_classes=N_CLASSES)