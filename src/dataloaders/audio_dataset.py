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
from helpers.audio_utils import *
from dataloaders.imbalanced_dataset_sampler import ImbalancedDatasetSampler

warnings.filterwarnings("ignore")

class AudioDataset(Dataset):
    def __init__(self, root_dir, csv_dir, conf, bird_code, mem_size=32, file_type="mp3", num_splits=5, apply_mix_aug = False, isTraining=True, transform=None):
        self.root_dir = root_dir
        self.data = list(pd.read_csv(csv_dir)[["filename", "ebird_code"]].to_dict('index').values())
        self.transform = transform
        self.conf = conf
        self.num_splits = num_splits
        self.isTraining = isTraining
        self.apply_mix_aug = apply_mix_aug
        self.bird_code = bird_code
        self.length = len(self.data)
        
        self.memory_buffer = []
        self.mem_size = mem_size
        self.file_type = file_type
        self.additional_loader_params = {
            "worker_init_fn": self.init_workers_fn
        }
        self.sampler = ImbalancedDatasetSampler

    def get_label(self, dataset, idx):
        return dataset.data[idx]["ebird_code"]

    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def __len__(self):
        return self.length
    
    def mix_aug(self, y_snippets, label):
        labels = [[label]] * len(y_snippets)
        mixed_snippets = y_snippets[:]
        
        for index, m_snip in enumerate(mixed_snippets):
            if len(self.memory_buffer)>self.mem_size*0.8:
                num_to_mix = np.random.randint(0,min(4, len(self.memory_buffer)))
                mem_y_snippets = [m_snip]
                mem_label_snippets = [label]
                for i in range(num_to_mix):
                    memory_item = self.memory_buffer.pop(0)
                    mem_y_snippets.append(memory_item["snippet"])
                    mem_label_snippets.append(memory_item["label"])
                    
                if len(mem_y_snippets)>1:
                    mixed_snippets[index]=np.array(mem_y_snippets).sum(0)/len(mem_y_snippets)
                    labels[index] = mem_label_snippets
                    
        
        if np.random.random()>0.1:
            if len(self.memory_buffer)<self.mem_size:
                mem_choice = np.random.randint(1,len(y_snippets)+1)
                choices_to_add = np.random.choice(list(range(0,len(y_snippets))), mem_choice, replace=False)
                for ch in choices_to_add:
                    self.memory_buffer.append({
                        "snippet": y_snippets[ch],
                        "label": label
                    })
        return mixed_snippets, labels
        

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item["ebird_code"]
        filename = item["filename"]
        
        if self.file_type=="mp3":
            file_dir = self.root_dir/f"{label}"/f"{filename}"
        else:
            file_dir = self.root_dir[label[0]]/f"{label}"/f"{filename.replace('.mp3','.wav')}"
            
            
        y, duration = read_audio(file_dir, self.conf)
        snip_duration = self.conf.sampling_rate*self.conf.duration
        if self.isTraining:
            if duration-snip_duration <= 0:
                indices = [0]*self.num_splits
            else:
                indices = np.random.randint(0,duration-snip_duration,self.num_splits)
        else:
            if (duration-snip_duration)<=0 or math.ceil((duration-snip_duration)/self.num_splits)<=0:
                indices = [0]*self.num_splits
            else:
                indices = list(range(0, duration-snip_duration,math.ceil((duration-snip_duration)/self.num_splits)))
        
        y_snippets = get_snippets(y, snip_duration, indices)

        if self.apply_mix_aug:
            y_snippets, list_of_labels = self.mix_aug(y_snippets, label)
        else:
            list_of_labels = [[label]] * len(y_snippets)
        
        list_of_images = []
        for y_snip in y_snippets:
            image = audio_to_melspectrogram(y_snip, self.conf)
            image = mono_to_color(image)
            list_of_images.append(image)
        
        coded_labels = np.zeros((len(list_of_images),len(self.bird_code)))
        for index, temp_labels in enumerate(list_of_labels):
            for temp_label in temp_labels:
                label_index = self.bird_code[temp_label]
                coded_labels[index][label_index] = 1
        
        list_of_images = np.array(list_of_images)
        if self.transform:
            list_of_images = self.transform(list_of_images)

        return {
            "filenames":filename, 
            "images": list_of_images,
            "coded_labels": torch.tensor(coded_labels).float()
        }