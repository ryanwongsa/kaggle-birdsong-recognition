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

class AudioDatasetV3(Dataset):
    def __init__(self, root_dir, csv_dir, conf, bird_code, inv_ebird_label, background_audio_dir=None, xeno_csv=None, xeno_dir=None, file_type="mp3", num_splits=5, apply_mixer = False, isTraining=True, transform=None):
        self.root_dir = root_dir
        df = pd.read_csv(csv_dir)
        df.secondary_labels = df.secondary_labels.apply(eval)
        df["xeno_source"] = False
        self.transform = transform
        self.conf = conf
        self.num_splits = num_splits
        self.isTraining = isTraining
        self.apply_mixer = apply_mixer
        self.bird_code = bird_code
        self.inv_ebird_label = inv_ebird_label

        self.file_type = file_type
        self.additional_loader_params = {
            "worker_init_fn": self.init_workers_fn
        }
        self.sampler = ImbalancedDatasetSampler
        
        if xeno_csv is not None:
            self.xeno_dir = xeno_dir
            df_xeno = pd.read_csv(xeno_csv)
            df_xeno.secondary_labels = df_xeno.secondary_labels.apply(eval)
            df_xeno["xeno_source"] = True
            df = pd.concat([df, df_xeno])
            df = df.reset_index(drop=True)
            del df_xeno

        if self.apply_mixer:
            ebird_counts = df.groupby("ebird_code").agg(len)
            ebird_counts["ratio"] = (len(df)-ebird_counts["filename"])/len(df)
            ebird_counts["prob"] = ebird_counts["ratio"]/ebird_counts["ratio"].sum()
            ebird_counts_dict = ebird_counts["prob"].to_dict()
            self.label_values, self.label_probs = list(ebird_counts_dict.keys()), list(ebird_counts_dict.values())
            self.dict_grp = {}
            for grp, d in df.groupby("ebird_code"):
                self.dict_grp[grp] = d.index.values

        self.data = list(df[["filename", "ebird_code", "secondary_labels", "xeno_source"]].to_dict('index').values())
        
        self.background_audio_dir = background_audio_dir
        if self.background_audio_dir is not None:
            for bk in background_audio_dir.glob('**/*.wav'):
                self.data.append({
                    "filename": bk,
                    "ebird_code": None
                })
        
        self.length = len(self.data)
        del df

    def get_label(self, dataset, idx):
        return dataset.data[idx]["ebird_code"]

    def init_workers_fn(self, worker_id):
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')
        np.random.seed(new_seed)

    def __len__(self):
        return self.length
    
    def mix_aug(self, y_snippets, original_labels):
        labels = [original_labels] * len(y_snippets)
        mixed_snippets = y_snippets[:]
        snip_duration = self.conf.sampling_rate*self.conf.duration
        for index, m_snip in enumerate(mixed_snippets):
            if np.random.random()>0.5:
                bird_type = np.random.choice(self.label_values, p=self.label_probs)
                index_choice = np.random.choice(self.dict_grp[bird_type])
                y_secondary, duration_secondary, labels_secondary, filename_secondary = self.get_audio(index_choice)
                if duration_secondary-snip_duration <= 0:
                    ix = 0
                else:
                    ix = np.random.randint(0,duration_secondary-snip_duration)
                y_secondary = y_secondary[ix:ix+snip_duration]
                mixed_snippets[index] = (m_snip + y_secondary)/2
                labels[index].extend(labels_secondary)

        return mixed_snippets, labels

    def get_audio(self, idx):
        item = self.data[idx]
        filename = item["filename"]
        if item["ebird_code"] is not None:
            is_xeno = item["xeno_source"]
            label = item["ebird_code"]
            label_names = item["secondary_labels"]
        
            added_label_codes = []
            for ln in label_names:
                if ln in self.inv_ebird_label:
                    added_label_codes.append(self.inv_ebird_label[ln])

            labels = added_label_codes+ [label]
        
            if self.file_type=="mp3":
                file_dir = self.root_dir/f"{label}"/f"{filename}"
            else:
                if is_xeno == False:
                    file_dir = self.root_dir[label[0]]/f"{label}"/f"{filename.replace('.mp3','.wav')}"
                else:
                    file_dir = self.xeno_dir/f"{label}"/f"{filename.replace('.mp3','.wav')}"
        else:
            labels = []
            file_dir = filename
            filename = filename.stem

        y, duration = read_audio(file_dir, self.conf)

        return y, duration, labels, filename

    def __getitem__(self, idx):
        y, duration, labels, filename = self.get_audio(idx)
        snip_duration = self.conf.sampling_rate*self.conf.duration
        self.isNoisy = False
        
        if self.isTraining:
            # if np.random.random()>0.2:
            #     noise_amp = np.random.uniform()*np.random.random()/np.random.choice([10,100])
            #     y = y + noise_amp * np.random.normal(size=y.shape[0])

            if duration-snip_duration <= 0:
                indices = [0]*self.num_splits
                self.isNoisy = False
                y_snippets = get_snippets(y, snip_duration, indices)
            else:
                if np.random.random()>0.2:
                    indices = np.random.randint(0,duration-snip_duration,self.num_splits)
                    self.isNoisy = True
                    y_snippets = get_snippets(y, snip_duration, indices)

                else:
                    self.isNoisy = False
                    y_snippets = []
                    for i in range(self.num_splits):
                        start_index = np.random.randint(0,self.conf.sampling_rate//10)
                        indices = range(start_index,duration-snip_duration,snip_duration)
                        if len(indices)==0:
                            indices = range(0,duration-snip_duration+1,snip_duration)
                        y_snippet = get_snippets(y, snip_duration, indices)
                        y_snippet = np.array(y_snippet)
                        y_snippet = y_snippet.mean(axis=0)
                        y_snippets.append(y_snippet)
                        
            if np.random.random()>0.75:
                num_divisions = np.random.choice([2,4,5])
                division_length = snip_duration//num_divisions
                selection_index = range(0,snip_duration,division_length)
                mixture_num = np.random.randint(1,num_divisions+1)
                
                for j in range(len(y_snippets)):
                    y_new = np.zeros(snip_duration)
                    for i in range(num_divisions):
                        temp = []
                        for ci in np.random.choice(selection_index,mixture_num, replace=False):
                            ci_y = y_snippets[j][ci:ci+division_length]
                            temp.append(ci_y)
                        temp = np.array(temp).mean(axis=0)
                        y_new[i*division_length:i*division_length+division_length] = temp
                    y_snippets[j] = y_new
        else:
            self.isNoisy = False
            y_snippets = []
            for i in range(self.num_splits):
                start_index = 0
                indices = range(start_index,duration-snip_duration,snip_duration)
                if len(indices)==0:
                    indices = range(0,duration-snip_duration+1,snip_duration)
                y_snippet = get_snippets(y, snip_duration, indices)
                y_snippet = np.array(y_snippet)
                y_snippet = y_snippet.mean(axis=0)
                y_snippets.append(y_snippet)
            
        if self.apply_mixer and self.isNoisy:
            y_snippets, list_of_labels = self.mix_aug(y_snippets, labels)
        else:
            list_of_labels = [labels] * len(y_snippets)
        
        if self.isNoisy:
            clean = [0] * len(y_snippets)
        else:
            clean = [1] * len(y_snippets)
            
        list_of_images = []
        for y_snip in y_snippets:
            image = audio_to_melspectrogram(y_snip, self.conf)
            list_of_images.append(image)
        
        coded_labels = np.zeros((len(list_of_images),len(self.bird_code)))
        for index, temp_labels in enumerate(list_of_labels):
            for temp_label in temp_labels:
                label_index = self.bird_code[temp_label]
                coded_labels[index][label_index] = 1
        
        if self.transform:
            list_of_images = self.transform(list_of_images)
        return {
            "filenames":filename, 
            "images": list_of_images,
            "coded_labels": torch.from_numpy(coded_labels).float(),
            "clean": torch.tensor(clean).float()
        }