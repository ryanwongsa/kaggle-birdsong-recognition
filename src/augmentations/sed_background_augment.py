from torchvision import transforms
import torch
import numpy as np
import librosa
import torch.nn.functional as F
import numpy as np
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, AddBackgroundNoise, AddShortNoises, Gain

def get_training_augmentation(augmenter): 
    def transform(x, sr):
        x = augmenter(samples=x, sample_rate=sr)
        return x
    return transform


def get_validation_augmentation():
    def transform(x, sr):
        return x
    return transform

def get_transforms(bckgrd_aug_dir=None, secondary_bckgrd_aug_dir=None):
    list_of_aug = [
#         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
        AddGaussianNoise(p=0.2),
        AddGaussianSNR(p=0.2),
        Gain(min_gain_in_db=-15,max_gain_in_db=15,p=0.3)
    ]
    if bckgrd_aug_dir is not None:
        list_of_aug.append(AddBackgroundNoise(bckgrd_aug_dir,p=0.2))
    if secondary_bckgrd_aug_dir is not None:
        list_of_aug.append(AddShortNoises(secondary_bckgrd_aug_dir,min_time_between_sounds=0.0, max_time_between_sounds=15.0, burst_probability=0.5, p=0.6))
    list_of_aug += [
        AddGaussianNoise(p=0.2),
        AddGaussianSNR(p=0.2),
        Gain(min_gain_in_db=-15,max_gain_in_db=15,p=0.3)
    ]
    augmenter = Compose(list_of_aug)
    transforms = {
        "train": get_training_augmentation(augmenter),
        "valid": get_validation_augmentation()
    }
    return transforms