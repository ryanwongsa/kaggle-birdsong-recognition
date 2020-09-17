from pathlib import Path
import torch
from config_params.configs import get_dict_value, BIRD_CODE, INV_EBIRD_LABEL
import os

class Parameters(object):
    def __init__(self, hparams=None):
        self.fold = 4
        self.name = os.path.basename(__file__).replace(".py","")
        
        self.aug_name = "secondary_default"
        self.apply_mixup = False
        self.model_name = "sed_dense121att"

        self.model_config =  {
            "sample_rate": 32000,
            "window_size": 1024,
            "hop_size": 320,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 14000,
            "classes_num": 264,
            "apply_aug": True,
            "top_db": None
        }
        self.pretrained_path = None

        self.bckgrd_aug_dir = "/pinknoise"
        self.secondary_bckgrd_aug_dir = "/pinknoise"

        self.optimizer_name = "adamw"
        self.weight_decay = 0.01

        self.criterion_name = "sed_scaled_pos_neg_focal_loss"
        self.criterion_params = {
            "gamma" : 0.0,
            "alpha_0" : 1.0,
            "alpha_1": 1.0,
            "secondary_factor": 1.0
        }
        

        self.scheduler_name = "warmup_with_cosine"
        self.lr_scale_factor = 0.01
        self.lr = 0.001

        self.logger_name = "neptune"
        
        self.PERIOD = 30

        self.train_ds_params = {
            "root_dir": Path("/data/"),
            "csv_dir": Path(f"/folds5/fold_{self.fold}_train.csv"),
            "period": self.PERIOD,
            "bird_code": BIRD_CODE,
            "inv_ebird_label":INV_EBIRD_LABEL,
            "isTraining": True,
            "num_test_samples": 1,
        }
        
        self.valid_ds_params = {
            "root_dir": Path("/data/"),
            "csv_dir": Path(f"/folds5/fold_{self.fold}_test.csv"),
            "period": self.PERIOD,
            "bird_code": BIRD_CODE,
            "inv_ebird_label":INV_EBIRD_LABEL,
            "background_audio_dir": Path("/background/data_ssw"),
            "isTraining": False,
            "num_test_samples": 2,
        }

        self.test_ds_params = {
            "root_dir": Path("/data/"),
            "csv_dir": Path(f"/folds5/fold_{self.fold}_test.csv"),
            "background_audio_dir":  Path("/background/data_ssw"),
            "period": self.PERIOD,
            "bird_code": BIRD_CODE,
            "inv_ebird_label":INV_EBIRD_LABEL,
            "isTraining": False,
            "num_test_samples": 2,
        }

        self.checkpoint_params = {
            "save_dir":f"/saved_models/{self.name}",
            "n_saved":10,
            "prefix_name":self.name,
        }
        
        self.train_bs = 28
        self.train_num_workers = 8
        self.valid_bs = 32
        self.valid_num_workers = 4
        self.metrics = ["lraps", "f1score_clip", "f1score_frame"]
        
        self.track_metric = "f1score_frame"
        self.metric_factor = 1
        
        self.checkpoint_dir = None
        self.add_pbar = True
        
        self.run_params = {
            "max_epochs": 50,
            "epoch_length": None
        }
        
        self.logger_params = {
            "project_name": "bird-song",
            "log_every": 10,
            "name": self.name,
            "prefix_name": f"{self.name}_best",
            "tags": [self.fold, self.name, self.model_name, self.criterion_name],
            "params": {
                "bs": self.train_bs,
                "lr": self.lr,
                "name": self.name,
                "aug_name": self.aug_name,
                "model_name": self.model_name,
                "weight_decay": self.weight_decay,
                "apply_mixup": self.apply_mixup,
                "optimizer_name": self.optimizer_name,
                "criterion_name": self.criterion_name,
                "scheduler_name": self.scheduler_name,
                "fold": self.fold,
                "lr_scale_factor": self.lr_scale_factor,
                "period": self.PERIOD,
                **self.model_config,
                **self.criterion_params
            }
        }
        
        self.dist_params = {
        }

        self.val_length = None
        self.eval_every = 2
        self.load_model_only = True
        self.accumulation_steps = 1
        self.gradient_clip_val = 0
