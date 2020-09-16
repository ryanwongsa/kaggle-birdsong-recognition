import torch
from pathlib import Path
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import os
import math
from PIL import Image
import warnings
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, AddBackgroundNoise

from helpers.audio_utils import *
from dataloaders.imbalanced_dataset_sampler import ImbalancedDatasetSampler

warnings.filterwarnings("ignore")

class AudioDatasetV4(Dataset):
    def __init__(self, root_dir, csv_dir, conf, bird_code,inv_ebird_label, num_test_samples=10, bckgrd_aug_dir=None, background_audio_dir=None, file_type="mp3", isTraining=True, transform=None, apply_mixer=False):
        self.root_dir = root_dir
        self.conf = conf
        self.isTraining = isTraining
        self.bird_code = bird_code
        self.inv_ebird_label = inv_ebird_label
        self.transform = transform
        self.file_type = file_type
        self.apply_mixer = apply_mixer
        self.additional_loader_params = {
            "worker_init_fn": self.init_workers_fn,
            "collate_fn": self.collate_fn
        }
        self.sampler = ImbalancedDatasetSampler
        
        df = pd.read_csv(csv_dir)
        df.secondary_labels = df.secondary_labels.apply(eval)
        self.data = list(df[["filename", "ebird_code", "secondary_labels"]].to_dict('index').values())
        
        self.background_audio_dir = background_audio_dir
        if self.background_audio_dir is not None:
            for bk in background_audio_dir.glob('**/*.wav'):
                self.data.append({
                    "filename": bk
                })
        
        
        self.num_test_samples = num_test_samples
        self.length = len(self.data)
        
        if self.apply_mixer:
            self.dict_grp = {}
            for grp, d in df.groupby("ebird_code"):
                self.dict_grp[grp] = d.index.values
            self.possible_mixer_keys = list(self.dict_grp.keys())
            
            if bckgrd_aug_dir is not None:
                self.augmenter = Compose([
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
                    AddGaussianSNR(p=0.3),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
                    AddBackgroundNoise(bckgrd_aug_dir,p=0.5),
                ])
            else:
                self.augmenter = Compose([
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
                    AddGaussianSNR(p=0.3),
                    PitchShift(min_semitones=-4, max_semitones=4, p=0.3)
                ])
        del df

    def get_label(self, dataset, idx):
        return dataset.data[idx]["ebird_code"]

    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)
        
    def collate_fn(self, samples):
        primary_codes = [s["primary_codes"] for s in samples]
        primary_codes = None if None in primary_codes else primary_codes
        return {
            "filenames": [s["filenames"] for s in samples],
            "images": torch.stack([s["images"] for s in samples],0),
            "coded_labels": torch.stack([s["coded_labels"] for s in samples],0),
            "primary_codes": primary_codes
        }

    def __len__(self):
        return self.length

    def get_audio(self, idx):
        item = self.data[idx]
        filename = item["filename"]
        if "ebird_code" in item:
            primary_label = item["ebird_code"]

            all_labels = [primary_label]
            for ln in item["secondary_labels"]:
                if ln in self.inv_ebird_label:
                    all_labels.append(self.inv_ebird_label[ln])

            if self.file_type=="mp3":
                file_dir = self.root_dir/f"{primary_label}"/f"{filename}"
            else:
                if type(self.root_dir) is dict:
                    file_dir = self.root_dir[primary_label[0]]/f"{primary_label}"/f"{filename.replace('.mp3','.wav')}"
                else:
                    file_dir = self.root_dir/f"{primary_label}"/f"{filename.replace('.mp3','.wav')}"
        else:
            primary_label = None
            all_labels = []
            file_dir = filename
            filename = filename.stem
            
            
        y, duration = read_audio(file_dir, self.conf)

        return {
            "y": y, 
            "duration": duration, 
            "primary_label":primary_label, 
            "all_labels":all_labels, 
            "filename": filename
        }
    
    def convert_labels_to_coded(self, num_images, labels):
        coded_labels = np.zeros((num_images,len(self.bird_code)))
        for index, temp_label in enumerate(labels):
            label_index = self.bird_code[temp_label]
            coded_labels[:,label_index] = 1
        return torch.from_numpy(coded_labels).float()
    
    def add_augmentation(self, y, primary_codes, all_labels):
        # if np.random.random()>0.5:
        #     bird_type = np.random.choice(self.possible_mixer_keys)
        #     index_choice = np.random.choice(self.dict_grp[bird_type])
        #     item2 = self.get_audio(index_choice)
            
        #     all_labels.extend(item2["all_labels"])
        #     ix = np.random.randint(0,item2["y"].shape[0]-self.conf.samples+1)
        #     y_secondary = item2["y"][ix:ix+self.conf.samples]
        #     y = (y + y_secondary)/2
        
        y = self.augmenter(samples=y, sample_rate=self.conf.sampling_rate)
        
        return y, primary_codes, all_labels

    def __getitem__(self, idx):
        item = self.get_audio(idx)
        
        if item["primary_label"] is not None:
            primary_codes = [self.bird_code[item["primary_label"]]]
        else:
            primary_codes = []
        all_labels = item["all_labels"]
        
        list_of_images = []
        if self.isTraining:
            for i in range(self.num_test_samples):
                start_index = np.random.randint(0,item["duration"]-self.conf.samples+1)
                end_index = start_index + self.conf.samples
                
                y_snippet = item["y"][start_index: end_index]
                y_snippet, primary_codes, all_labels = self.add_augmentation(y_snippet, primary_codes, all_labels)
                
                spectrogram = audio_to_melspectrogram(y_snippet, self.conf)
                list_of_images.append(spectrogram)
            
            primary_codes = primary_codes * len(list_of_images) 
        else:
            split_hop = math.ceil((item["duration"]-self.conf.samples)/self.num_test_samples)
            total_duration = item["duration"]-self.conf.samples
            if split_hop > 0 and total_duration > 0:
                for start_index in range(0,total_duration,split_hop):
                    end_index = start_index + self.conf.samples
                    y_snippet = item["y"][start_index: end_index]
                    
                    spectrogram = audio_to_melspectrogram(y_snippet, self.conf)
                    list_of_images.append(spectrogram)
            
            while(len(list_of_images)<self.num_test_samples):
                start_index = 0
                end_index = start_index + self.conf.samples
                y_snippet = item["y"][start_index: end_index]
                spectrogram = audio_to_melspectrogram(y_snippet, self.conf)
                list_of_images.append(spectrogram)
                
            primary_codes = None
        
        coded_labels = self.convert_labels_to_coded(len(list_of_images),all_labels)
        if self.transform:
            list_of_images = self.transform(list_of_images)
        return {
            "filenames":item["filename"], 
            "images": list_of_images,
            "coded_labels": coded_labels,
            "primary_codes": primary_codes
        }