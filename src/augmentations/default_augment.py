from torchvision import transforms
import torch
import numpy as np
import librosa
import torch.nn.functional as F
import numpy as np

def random_crop(x):
    if np.random.random()>0.66:
        c, x_x_shape, x_y_y_shape = x.shape
        x_crop_1 = np.random.randint(1,10)
        x_crop_2 = np.random.randint(1,10)
        y_crop_1 = np.random.randint(1,25)
        y_crop_2 = np.random.randint(1,25)
        x = x[:,x_crop_1:x_x_shape-x_crop_2, y_crop_1:x_y_y_shape-y_crop_2]
    return x

def get_training_augmentation(): 
    def transform(x):
        x_new = []
        for i, x_i in enumerate(x):         
            delta = librosa.feature.delta(x_i)
            accelerate = librosa.feature.delta(x_i, order=2)
            x_i = np.stack([x_i, delta, accelerate], axis=0)
            x_i = x_i / 30
            
            x_i = random_crop(x_i)
            x_i = torch.from_numpy(x_i).unsqueeze(0)
            x_i = F.interpolate(x_i, size=(256,256))#,recompute_scale_factor=True)
            
            x_new.append(x_i)
        return torch.cat(x_new,0).float()
    return transform


def get_validation_augmentation():
    def transform(x):
        x_new = []
        for i, x_i in enumerate(x):
            delta = librosa.feature.delta(x_i)
            accelerate = librosa.feature.delta(x_i, order=2)
            x_i = np.stack([x_i, delta, accelerate], axis=0)
            x_i = x_i / 30
            x_i = torch.from_numpy(x_i).unsqueeze(0)
            x_i = F.interpolate(x_i, size=(256,256))#,recompute_scale_factor=True)
            x_new.append(x_i)
        return torch.cat(x_new,0).float()
    return transform

def get_transforms():
    transforms = {
        "train": get_training_augmentation(),
        "valid": get_validation_augmentation()
    }
    return transforms