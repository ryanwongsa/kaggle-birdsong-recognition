from torchvision import transforms
import torch

def get_training_augmentation():
    def transform(x):
        return torch.tensor(x/255).permute(0,3, 1, 2).float()
    return transform


def get_validation_augmentation():
    def transform(x):
        return torch.tensor(x/255).permute(0,3, 1, 2).float()
    return transform

def get_transforms():
    transforms = {
        "train": get_training_augmentation(),
        "valid": get_validation_augmentation()
    }
    return transforms