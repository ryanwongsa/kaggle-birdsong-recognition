from torchvision import transforms
import torch
import numpy as np
import librosa
import torch.nn.functional as F
import numpy as np
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, AddBackgroundNoise

def get_training_augmentation(augmenter): 
    def transform(x, sr):
        x = augmenter(samples=x, sample_rate=sr)
        return x
    return transform


def get_validation_augmentation():
    def transform(x, sr):
        return x
    return transform

def get_transforms(bckgrd_aug_dir=None):
    list_of_aug = [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
        AddGaussianSNR(p=0.3)
    ]
    if bckgrd_aug_dir is not None:
        list_of_aug.append(AddBackgroundNoise(bckgrd_aug_dir,p=0.5))
    augmenter = Compose(list_of_aug)
    transforms = {
        "train": get_training_augmentation(augmenter),
        "valid": get_validation_augmentation()
    }
    return transforms