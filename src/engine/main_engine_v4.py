from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import os
import cProfile
from ignite.engine import Events, Engine
from ignite.handlers import Checkpoint
from engine.base.base_engine import BaseEngine
from ignite.utils import convert_tensor

class MainEngineV4(BaseEngine):
    def __init__(self, local_rank, hparams):
        super().__init__(local_rank, hparams)
    
    def prepare_batch(self, batch, mode = 'valid'):
        if mode == 'train':
            x, y = batch["images"], (batch["coded_labels"], batch["primary_codes"])
            return (
                convert_tensor(x, device=self.device, non_blocking=True),
                (
                    convert_tensor(y[0], device=self.device, non_blocking=True),
                    y[1]
                )
            )
        elif mode == 'valid':
            x, y = batch["images"], (batch["coded_labels"], batch["primary_codes"])
            return (
                convert_tensor(x, device=self.device, non_blocking=True),
                (
                    convert_tensor(y[0], device=self.device, non_blocking=True),
                    y[1]
                )
            )
        elif mode == 'test':
            x, inputs = batch["images"], batch
            return (
                convert_tensor(x, device=self.device, non_blocking=True),
                (inputs)
            )
        
    
    def loss_fn(self, y_pred, y):
        loss_factors = self.hparams.loss_factors # [1.0, 0.4,0.2,0.1]
        betas = self.hparams.betas #  [0.6,0.7,0.8,0.9]
        total_loss = 0.0
        component_loss = {}
        for i, y_pred_i in enumerate(y_pred):
            loss, dict_loss = self.criterion(y_pred_i, y, betas[i], betas[i]+0.05)
            total_loss += loss * loss_factors[i]
            for key, value in dict_loss.items():
                component_loss[f"{key}_{i}"] = value.detach()
        return total_loss, component_loss
    
    def output_transform(self, x, y, y_pred, loss=None, dict_loss={}, mode = 'valid'):
        if mode == 'train':
            return {"loss": loss.detach(), "x": x, "y_pred": y_pred, "y":y[0], "dict_loss": dict_loss}
        elif mode == 'valid':
            return {"loss": loss.detach(), "x": x, "y_pred": y_pred, "y":y[0], "dict_loss": dict_loss}
        elif mode == 'test':
            return {"y_pred": y_pred, "x": x, "input":y}

    def _init_optimizer(self):
        if self.hparams.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer_name == "adamw_no_weight_decay":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0.0)
    
    def _init_criterion_function(self):
        if self.hparams.criterion_name == "noisy_controlled_loss":
            from loss.noisy_controlled_loss import NoisyControlledLoss
            self.criterion = NoisyControlledLoss()
        elif self.hparams.criterion_name == "noisy_loss":
            from loss.noisy_loss import NoisyLoss
            self.criterion = NoisyLoss()
        elif self.hparams.criterion_name == "f1_controlled_loss":
            from loss.f1_controlled_loss import F1ControlledLoss
            self.criterion = F1ControlledLoss()
        elif self.hparams.criterion_name == "bce_controlled_loss":
            from loss.bce_controlled_loss import BCEControlledLoss
            self.criterion = BCEControlledLoss(primary_loss_factor = self.hparams.primary_loss_factor)
        elif self.hparams.criterion_name == "bce_controlled_standard_loss":
            from loss.bce_controlled_standard_loss import BCEControlledStandardLoss
            self.criterion = BCEControlledStandardLoss()
        elif self.hparams.criterion_name == "focal_controlled_standard_loss":
            from loss.focal_controlled_standard_loss import FocalControlledStandardLoss
            self.criterion = FocalControlledStandardLoss(gamma=self.hparams.gamma, alpha=self.hparams.alpha)
        elif self.hparams.criterion_name == "focal_mse_loss":
            from loss.focal_mse_loss import FocalMSELoss
            self.criterion = FocalMSELoss(gamma=self.hparams.gamma, alpha=self.hparams.alpha)

    def _init_scheduler(self):
        if self.hparams.scheduler_name == "none":
            self.scheduler = None
        elif self.hparams.scheduler_name == "warmup_with_cosine":
            from ignite.contrib.handlers import LinearCyclicalScheduler, CosineAnnealingScheduler, ConcatScheduler
            lr = self.hparams.lr
            if self.hparams.run_params["epoch_length"]:
                epoch_length = self.hparams.run_params["epoch_length"]
            else:
                epoch_length = len(self.train_loader)
            num_epochs = self.hparams.run_params["max_epochs"]
            scheduler_1 = LinearCyclicalScheduler(self.optimizer, "lr", start_value=lr*0.01, end_value=lr, cycle_size=epoch_length*2)
            scheduler_2 = CosineAnnealingScheduler(self.optimizer, "lr", start_value=lr, end_value=lr*0.001, cycle_size=num_epochs*epoch_length)
            durations = [epoch_length, ]
            self.scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=durations)
        elif self.hparams.scheduler_name == "warmup_with_cosine_100":
            from ignite.contrib.handlers import LinearCyclicalScheduler, CosineAnnealingScheduler, ConcatScheduler
            lr = self.hparams.lr
            if self.hparams.run_params["epoch_length"]:
                epoch_length = self.hparams.run_params["epoch_length"]
            else:
                epoch_length = len(self.train_loader)
            num_epochs = self.hparams.run_params["max_epochs"]
            scheduler_1 = LinearCyclicalScheduler(self.optimizer, "lr", start_value=lr*0.01, end_value=lr, cycle_size=epoch_length*2)
            scheduler_2 = CosineAnnealingScheduler(self.optimizer, "lr", start_value=lr, end_value=lr*0.01, cycle_size=num_epochs*epoch_length)
            durations = [epoch_length, ]
            self.scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=durations)
        elif self.hparams.scheduler_name == "warmup_with_cosine_10":
            from ignite.contrib.handlers import LinearCyclicalScheduler, CosineAnnealingScheduler, ConcatScheduler
            lr = self.hparams.lr
            if self.hparams.run_params["epoch_length"]:
                epoch_length = self.hparams.run_params["epoch_length"]
            else:
                epoch_length = len(self.train_loader)
            num_epochs = self.hparams.run_params["max_epochs"]
            scheduler_1 = LinearCyclicalScheduler(self.optimizer, "lr", start_value=lr*0.1, end_value=lr, cycle_size=epoch_length*2)
            scheduler_2 = CosineAnnealingScheduler(self.optimizer, "lr", start_value=lr, end_value=lr*0.1, cycle_size=num_epochs*epoch_length)
            durations = [epoch_length, ]
            self.scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=durations)
        elif self.hparams.scheduler_name == "one_cycle_cosine_10":
            from ignite.contrib.handlers import CosineAnnealingScheduler
            lr = self.hparams.lr
            if self.hparams.run_params["epoch_length"]:
                epoch_length = self.hparams.run_params["epoch_length"]
            else:
                epoch_length = len(self.train_loader)
            num_epochs = self.hparams.run_params["max_epochs"]
            self.scheduler  = CosineAnnealingScheduler(self.optimizer, "lr", start_value=lr, end_value=lr*0.1, cycle_size=num_epochs*epoch_length)
        elif self.hparams.scheduler_name == "one_cycle_cosine_100":
            from ignite.contrib.handlers import CosineAnnealingScheduler
            lr = self.hparams.lr
            if self.hparams.run_params["epoch_length"]:
                epoch_length = self.hparams.run_params["epoch_length"]
            else:
                epoch_length = len(self.train_loader)
            num_epochs = self.hparams.run_params["max_epochs"]
            self.scheduler  = CosineAnnealingScheduler(self.optimizer, "lr", start_value=lr, end_value=lr*0.01, cycle_size=num_epochs*epoch_length)
            
    def _init_logger(self):
        if self.hparams.logger_name == "print":
            from logger.print.print_logger import PrintLogger
            self.logger = PrintLogger(**self.hparams.logger_params)
        elif self.hparams.logger_name == "neptune":
            from logger.neptune.neptune_logger import MyNeptuneLogger
            self.logger = MyNeptuneLogger(**self.hparams.logger_params)
        else:
            self.logger = None

    def _init_metrics(self):
        from ignite.metrics import Loss, RunningAverage
        
        self.train_metrics = {
            'train_avg_loss': RunningAverage(output_transform=lambda x: x["loss"])
        }

        self.validation_metrics = {
            'valid_avg_loss': RunningAverage(output_transform=lambda x: x["loss"])
        }

        def add_criterion_metrics(loss_key, i):
            self.train_metrics[f"train_{loss_key}_{i}"] = RunningAverage(output_transform=lambda x: x["dict_loss"][f"{loss_key}_{i}"])
            self.validation_metrics[f"valid_{loss_key}_{i}"] = RunningAverage(output_transform=lambda x: x["dict_loss"][f"{loss_key}_{i}"])

        for lk in self.criterion.loss_keys:
            for i in range(4):
                add_criterion_metrics(lk, i)

        self.test_metrics = {}

        if "f1score50" in self.hparams.metrics:
            from metrics.custom_f1score import CustomF1Score
            for i in range(1):
                self.validation_metrics[f"f1score50_{i}"] = CustomF1Score(threshold=0.5, output_transform=lambda x: (x["y_pred"][i], x["y"]))
        
        if "f1score60" in self.hparams.metrics:
            from metrics.custom_f1score import CustomF1Score
            for i in range(1):
                self.validation_metrics[f"f1score60_{i}"] = CustomF1Score(threshold=0.6, output_transform=lambda x: (x["y_pred"][i], x["y"]))

        if "f1score70" in self.hparams.metrics:
            from metrics.custom_f1score import CustomF1Score
            for i in range(1):
                self.validation_metrics[f"f1score70_{i}"] = CustomF1Score(threshold=0.7, output_transform=lambda x: (x["y_pred"][i], x["y"]))

        if "test" in self.hparams.metrics:
            from metrics.custom_test import CustomTest
            self.test_metrics["test"] = CustomTest(save_dir=self.hparams.save_dir, combine_files=self.hparams.combine_files, output_transform=lambda x: (x["y_pred"][0], x["input"]))

    def _init_model(self):
        if self.hparams.model_name == "aux_dense121":
            from models.aux_densenet121 import AUXDense121
            self.model = AUXDense121(num_classes = self.hparams.num_classes)
    
    def _init_augmentation(self):
        if self.hparams.aug_name == "default":
            from augmentations.default_augment import get_transforms
            self.tfms = get_transforms()
        
    def _init_train_datalader(self):
        from dataloaders.audio_dataset_v4 import AudioDatasetV4
        self.train_ds = AudioDatasetV4(**self.hparams.train_ds_params, transform=self.tfms["train"])

    def _init_valid_dataloader(self):
        from dataloaders.audio_dataset_v4 import AudioDatasetV4
        self.valid_ds = AudioDatasetV4(**self.hparams.valid_ds_params, transform=self.tfms["valid"])

    def _init_test_dataloader(self):
        from dataloaders.audio_dataset_v4 import AudioDatasetV4
        self.test_ds = AudioDatasetV4(**self.hparams.test_ds_params, transform=self.tfms["valid"])
            
        
