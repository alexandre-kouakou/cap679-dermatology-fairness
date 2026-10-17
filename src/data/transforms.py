from torchvision import transforms
from src.config import IMAGENET_MEAN, IMAGENET_STD

def get_transforms(train: bool) -> transforms.Compose:
    """Training pipeline augments; evaluation pipeline only resizes.

    Fairness note: brightness/contrast jitter is kept mild and hue/saturation
    are deliberately untouched, so skin tone is not altered by augmentation.
    """
    if train:
        return transforms.Compose([
            transforms.RandomRotation(degrees = 30),
            transforms.RandomResizedCrop(224, scale = (0.7, 1.0)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.ColorJitter(brightness = 0.1, contrast = 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
    ])