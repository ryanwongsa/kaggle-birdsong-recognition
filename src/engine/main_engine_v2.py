from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import os
import cProfile
from ignite.engine import Events, Engine
from ignite.handlers import Checkpoint
from engine.base.base_engine import BaseEngine
from ignite.utils import convert_tensor

class MainEngineV2(BaseEngine):
    def __init__(self, local_rank, hparams):
        super().__init__(local_rank, hparams)
    
    def prepare_batch(self, batch, mode = 'valid'):
        if mode == 'train':
            x, y = batch["images"], (batch["coded_labels"], batch["clean"])
        elif mode == 'valid':
            x, y = batch["images"], (batch["coded_labels"], batch["clean"])
        elif mode == 'test':
            x, inputs = batch["images"], batch
            return (
                convert_tensor(x, device=self.device, non_blocking=True),
                (inputs)
            )
        return (
            convert_tensor(x, device=self.device, non_blocking=True),
            (
                convert_tensor(y[0], device=self.device, non_blocking=True),
                convert_tensor(y[1], device=self.device, non_blocking=True)
            )
        )
    
    def loss_fn(self, y_pred, y):
        loss, dict_loss = self.ls_fn(y_pred, y)
        return loss, dict_loss
    
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
        elif self.hparams.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
    
    def _init_criterion_function(self):
        if self.hparams.criterion_name == "bce":
            from loss.bce_loss import BCELoss
            self.criterion = BCELoss()
        elif self.hparams.criterion_name == "smooth_bce":
            from loss.smooth_bce_loss import SmoothBCELoss
            self.criterion = SmoothBCELoss(smooth=self.hparams.smooth)
        elif self.hparams.criterion_name in ["lq", "lsoft"]:
            from loss.custom_losses import NoisyCuratedLoss
            self.criterion = NoisyCuratedLoss(noisy_type=self.hparams.criterion_name, beta=self.hparams.beta, q=self.hparams.q) 
        elif self.hparams.criterion_name in ["lsoft_targetted"]:
            from loss.custom_losses_for_targets import NoisyCuratedLossTargetted
            self.criterion = NoisyCuratedLossTargetted(noisy_type=self.hparams.criterion_name, beta=self.hparams.beta, q=self.hparams.q) 

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

        self.test_metrics = {}

        if "f1score50" in self.hparams.metrics:
            from metrics.custom_f1score import CustomF1Score
            self.validation_metrics["f1score50"] = CustomF1Score(threshold=0.5, output_transform=lambda x: (x["y_pred"], x["y"]))
        
        if "f1score60" in self.hparams.metrics:
            from metrics.custom_f1score import CustomF1Score
            self.validation_metrics["f1score60"] = CustomF1Score(threshold=0.6, output_transform=lambda x: (x["y_pred"], x["y"]))

        if "f1score70" in self.hparams.metrics:
            from metrics.custom_f1score import CustomF1Score
            self.validation_metrics["f1score70"] = CustomF1Score(threshold=0.7, output_transform=lambda x: (x["y_pred"], x["y"]))

        if "lraps" in self.hparams.metrics:
            from metrics.custom_lraps import CustomLRAPS
            self.validation_metrics["lraps"] = CustomLRAPS(output_transform=lambda x: (x["y_pred"], x["y"]))

        if "test" in self.hparams.metrics:
            from metrics.custom_test import CustomTest
            self.test_metrics["test"] = CustomTest(save_dir=self.hparams.save_dir, combine_files=self.hparams.combine_files, output_transform=lambda x: (x["y_pred"], x["input"]))

    def _init_model(self):
        if self.hparams.model_name == "dcase":
            from models.classifier_dcase import Classifier_DCase
            self.model = Classifier_DCase(self.hparams.num_classes)
        elif self.hparams.model_name == "densenet121":
            from models.densenet121 import DenseNet121
            self.model = DenseNet121(num_classes=self.hparams.num_classes)
        elif self.hparams.model_name == "densenet161":
            from models.densenet161 import DenseNet161
            self.model = DenseNet161(num_classes=self.hparams.num_classes)
    
    def _init_augmentation(self):
        if self.hparams.aug_name == "baseline":
            from augmentations.base_augment import get_transforms
            self.tfms = get_transforms()
        elif self.hparams.aug_name == "spec_aug":
            from augmentations.spec_augment import get_transforms
            self.tfms = get_transforms()
        
    def _init_train_datalader(self):
        from dataloaders.audio_dataset_v2 import AudioDatasetV2
        self.train_ds = AudioDatasetV2(**self.hparams.train_ds_params, transform=self.tfms["train"])

    def _init_valid_dataloader(self):
        from dataloaders.audio_dataset_v2 import AudioDatasetV2
        self.valid_ds = AudioDatasetV2(**self.hparams.valid_ds_params, transform=self.tfms["valid"])

    def _init_test_dataloader(self):
        from dataloaders.audio_dataset_v2 import AudioDatasetV2
        self.test_ds = AudioDatasetV2(**self.hparams.test_ds_params, transform=self.tfms["valid"])
            
        
