from torchvision import transforms
import torch
import numpy as np
import librosa
import cv2
from skimage.transform import resize
import torch.nn.functional as F
from helpers.audio_utils import mono_to_color
def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = np.random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num  = spec.shape
        freq_percentage = np.random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = np.random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec

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
            if np.random.random()>0.5:
                x_i = spec_augment(x_i, value=x_i.min())
            x_i = mono_to_color(x_i)/255.0
            x_i = random_crop(x_i)
            x_i = torch.from_numpy(x_i).unsqueeze(0)
            x_i = F.interpolate(x_i, size=(256,512))
            
            x_new.append(x_i)
        return torch.cat(x_new,0).float()
    return transform


def get_validation_augmentation():
    def transform(x):
        x_new = []
        for i, x_i in enumerate(x):
            x_i = mono_to_color(x_i)/255.0
            x_i = torch.from_numpy(x_i).unsqueeze(0)
            x_i = F.interpolate(x_i, size=(256,512))
            x_new.append(x_i)
        return torch.cat(x_new,0).float()
    return transform

def get_transforms():
    transforms = {
        "train": get_training_augmentation(),
        "valid": get_validation_augmentation()
    }
    return transforms