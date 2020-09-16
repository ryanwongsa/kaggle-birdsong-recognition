from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
import pandas as pd
import torch
import math
import os
from dataloaders.imbalanced_dataset_sampler import ImbalancedDatasetSampler

class SedDataset(Dataset):
    def __init__( self,
            root_dir, csv_dir, period, bird_code, inv_ebird_label, isTraining,num_test_samples, transform=None, background_audio_dir=None):

        self.root_dir = root_dir
        self.transform = transform
        self.period = period
        self.bird_code = bird_code
        self.inv_ebird_label = inv_ebird_label
        self.isTraining = isTraining
        self.num_test_samples = num_test_samples
        
        self.additional_loader_params = {
            "worker_init_fn": self.init_workers_fn
        }
        self.sampler = ImbalancedDatasetSampler
        
        df = pd.read_csv(csv_dir)
        df.secondary_labels = df.secondary_labels.apply(eval)
        self.data = list(df[["filename", "ebird_code", "secondary_labels"]].to_dict('index').values())
        del df
        
        self.background_audio_dir = background_audio_dir
        has_background = False
        if self.background_audio_dir is not None:
            for bk in background_audio_dir.glob('**/*.wav'):
                self.data.append({
                    "filename": bk
                })
                has_background =True
        
        if has_background:
            print("Background audio was added to dataloader")
        self.length = len(self.data)
    
    def get_label(self, dataset, idx):
        return dataset.data[idx]["ebird_code"]

    def __len__(self):
        return self.length

    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)
        
    def get_audio(self, idx):
        item = self.data[idx]
        filename = item["filename"]
        if "ebird_code" in item:
            primary_label = item["ebird_code"]

            all_labels = [primary_label]
            for ln in item["secondary_labels"]:
                if ln in self.inv_ebird_label:
                    all_labels.append(self.inv_ebird_label[ln])

            if type(self.root_dir) is dict:
                file_dir = self.root_dir[primary_label[0]]/f"{primary_label}"/f"{filename.replace('.mp3','.wav')}"
            else:
                file_dir = self.root_dir/f"{primary_label}"/f"{filename.replace('.mp3','.wav')}"
        else:
            primary_label = None
            all_labels = []
            file_dir = filename
            filename = filename.stem

        y, sr = sf.read(file_dir)
        return {
            "y": y, 
            "all_labels":all_labels, 
            "primary_labels": [primary_label] if primary_label is not None else [],
            "secondary_labels": [x for x in all_labels if x != primary_label],
            "sr": sr,
            "duration": len(y),
            "filename": filename
        }
    
    def convert_labels_to_coded(self, num_images, labels):
        coded_labels = np.zeros((num_images,len(self.bird_code)))
        for index, temp_label in enumerate(labels):
            label_index = self.bird_code[temp_label]
            coded_labels[:,label_index] = 1
        return torch.from_numpy(coded_labels).float()
    
    def __getitem__(self, idx: int):
        item = self.get_audio(idx)
        y = item["y"]
        
        if self.transform:
            y = self.transform(y, item["sr"])

        effective_length = item["sr"] * self.period
        list_of_y = []
        if self.isTraining:
            for i in range(self.num_test_samples):
                if item["duration"] < effective_length:
                    new_y = np.zeros(effective_length, dtype=y.dtype)
                    start = np.random.randint(effective_length - item["duration"])
                    new_y[start:start + item["duration"]] = y
                    y_snippet = new_y.astype(np.float32)
                elif item["duration"] > effective_length:
                    start = np.random.randint(item["duration"] - effective_length)
                    y_snippet = y[start:start + effective_length].astype(np.float32)
                else:
                    y_snippet = y.astype(np.float32)
                list_of_y.append(y_snippet)
        else:
            if item["duration"] < effective_length*self.num_test_samples:
                new_y = np.zeros(effective_length*self.num_test_samples, dtype=y.dtype)
                new_y[0:item["duration"]] = y
                y = new_y.astype(np.float32)
                item["duration"] = len(y)

            for i in range(self.num_test_samples):
                y_snippet = y[effective_length*i: effective_length*i+effective_length].astype(np.float32)
                list_of_y.append(y_snippet)
            
            
        all_labels = self.convert_labels_to_coded(len(list_of_y), item["all_labels"])
        primary_labels = self.convert_labels_to_coded(len(list_of_y), item["primary_labels"])
        secondary_labels = self.convert_labels_to_coded(len(list_of_y), item["secondary_labels"])

        list_of_y = np.array(list_of_y)
        return {"filename":item["filename"], "waveforms": list_of_y, "all_labels": all_labels, "primary_labels":primary_labels, "secondary_labels": secondary_labels}