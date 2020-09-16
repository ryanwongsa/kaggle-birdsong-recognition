from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import os
import cProfile
from ignite.engine import Events, Engine
from ignite.handlers import Checkpoint
from engine.base.base_engine import BaseEngine
from ignite.utils import convert_tensor
from augmentations.mixup import do_mixup

class SedEngine(BaseEngine):
    def __init__(self, local_rank, hparams):
        super().__init__(local_rank, hparams)
    
    def prepare_batch(self, batch, mode = 'valid'):
        if mode == 'train':
            x = convert_tensor(batch["waveforms"], device=self.device, non_blocking=True)
            bs, c, s = x.shape 

            all_labels = convert_tensor(batch["all_labels"], device=self.device, non_blocking=True)
            primary_labels = convert_tensor(batch["primary_labels"], device=self.device, non_blocking=True)
            secondary_labels = convert_tensor(batch["secondary_labels"], device=self.device, non_blocking=True)

            if self.mixup_augmenter is not None:
                mixup_lambda = self.mixup_augmenter.get_lambda(batch_size=bs*c, device=x.device)
                bs_al, s_al, c_al = all_labels.shape
                all_labels = do_mixup(all_labels.reshape(bs_al*s_al,c_al), mixup_lambda).reshape((bs_al*s_al)//2,1,c_al)
                primary_labels = do_mixup(primary_labels.reshape(bs_al*s_al,c_al), mixup_lambda).reshape((bs_al*s_al)//2,1,c_al)
                secondary_labels = do_mixup(secondary_labels.reshape(bs_al*s_al,c_al), mixup_lambda).reshape((bs_al*s_al)//2,1,c_al)
            else:
                mixup_lambda = None
                
            return (
                (
                    x,
                    mixup_lambda
                ),
                {
                    "all_labels": all_labels,
                    "primary_labels": primary_labels,
                    "secondary_labels": secondary_labels,
                }
            )
        elif mode == 'valid':
            return (
                (
                    convert_tensor(batch["waveforms"], device=self.device, non_blocking=True),
                    None
                ),
                {
                    "all_labels": convert_tensor(batch["all_labels"], device=self.device, non_blocking=True),
                    "primary_labels": convert_tensor(batch["primary_labels"], device=self.device, non_blocking=True),
                    "secondary_labels": convert_tensor(batch["secondary_labels"], device=self.device, non_blocking=True),
                }
            )
        elif mode == 'test':
            return (
                (
                    convert_tensor(x, device=self.device, non_blocking=True),
                    None
                ),
                {
                    "all_labels": convert_tensor(batch["all_labels"], device=self.device, non_blocking=True),
                    "primary_labels": convert_tensor(batch["primary_labels"], device=self.device, non_blocking=True),
                    "secondary_labels": convert_tensor(batch["secondary_labels"], device=self.device, non_blocking=True),
                    "filename": batch["filename"]
                }
            )
    
    def loss_fn(self, y_pred, y):
        loss, dict_loss = self.criterion(y_pred["clipwise_output"], y)
        return loss, dict_loss
    
    def output_transform(self, x, y, y_pred, loss=None, dict_loss={}, mode = 'valid'):
        if mode == 'train':
            return {"loss": loss.detach(), "x": x, "y_pred": y_pred, "y":y, "dict_loss": dict_loss}
        elif mode == 'valid':
            return {"loss": loss.detach(), "x": x, "y_pred": y_pred, "y":y, "dict_loss": dict_loss}
        elif mode == 'test':
            return {"y_pred": y_pred, "x": x, "y":y}

    def _init_optimizer(self):
        if self.hparams.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
    
    def _init_criterion_function(self):
        if self.hparams.criterion_name == "focal_loss_standard":
            from loss.focal_loss_standard import FocalLossStandard
            self.criterion = FocalLossStandard(**self.hparams.criterion_params)
        elif self.hparams.criterion_name == "sed_removed_focal_loss":
            from loss.sed_removed_focal_loss import SedRemovedFocalLoss
            self.criterion = SedRemovedFocalLoss (**self.hparams.criterion_params)
        elif self.hparams.criterion_name == "sed_scaled_pos_neg_focal_loss":
            from loss.sed_scaled_pos_neg_focal_loss import SedScaledPosNegFocalLoss
            self.criterion = SedScaledPosNegFocalLoss(**self.hparams.criterion_params)
        elif self.hparams.criterion_name == "sed_scaled_pos_neg_focal_loss_augd":
            from loss.sed_scaled_pos_neg_focal_loss_augd import SedScaledPosNegFocalLossAugd
            self.criterion = SedScaledPosNegFocalLossAugd(**self.hparams.criterion_params)
        elif self.hparams.criterion_name == "bce_controlled_standard_loss":
            from loss.bce_controlled_standard_loss import BCEControlledStandardLoss
            self.criterion = BCEControlledStandardLoss(**self.hparams.criterion_params)

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
            scheduler_1 = LinearCyclicalScheduler(self.optimizer, "lr", start_value=lr*self.hparams.lr_scale_factor, end_value=lr, cycle_size=epoch_length*2)
            scheduler_2 = CosineAnnealingScheduler(self.optimizer, "lr", start_value=lr, end_value=lr*self.hparams.lr_scale_factor, cycle_size=num_epochs*epoch_length)
            durations = [epoch_length, ]
            self.scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=durations)
        
        elif self.hparams.scheduler_name == "one_cycle_cosine":
            from ignite.contrib.handlers import CosineAnnealingScheduler
            lr = self.hparams.lr
            if self.hparams.run_params["epoch_length"]:
                epoch_length = self.hparams.run_params["epoch_length"]
            else:
                epoch_length = len(self.train_loader)
            num_epochs = self.hparams.run_params["max_epochs"]
            self.scheduler  = CosineAnnealingScheduler(self.optimizer, "lr", start_value=lr, end_value=lr*self.hparams.lr_scale_factor, cycle_size=num_epochs*epoch_length)

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

        def add_criterion_metrics(loss_key):
            self.train_metrics[f"train_{loss_key}"] = RunningAverage(output_transform=lambda x: x["dict_loss"][f"{loss_key}"])
            self.validation_metrics[f"valid_{loss_key}"] = RunningAverage(output_transform=lambda x: x["dict_loss"][f"{loss_key}"])

        for lk in self.criterion.loss_keys:
            add_criterion_metrics(lk)

        self.test_metrics = {}

        if "lraps" in self.hparams.metrics:
            from metrics.sed_lraps import SedLRAPS
            self.validation_metrics["lraps"] = SedLRAPS(output_transform=lambda x: (x["y_pred"], x["y"]["all_labels"]))

        if "f1score_clip" in self.hparams.metrics:
            from metrics.sed_f1score_clip import SedF1ScoreClip
            self.validation_metrics[f"f1score_clip"] = SedF1ScoreClip(threshold=0.5, output_transform=lambda x: (x["y_pred"], x["y"]["all_labels"]))

        if "f1score_frame" in self.hparams.metrics:
            from metrics.sed_f1score_frame import SedF1ScoreFrame
            self.validation_metrics[f"f1score_frame"] = SedF1ScoreFrame(threshold=0.5, output_transform=lambda x: (x["y_pred"], x["y"]["all_labels"]))

        if "test" in self.hparams.metrics:
            from metrics.sed_test import SedTest
            self.test_metrics["test"] = SedTest(save_dir=self.hparams.save_dir, combine_files=self.hparams.combine_files, output_transform=lambda x: (x["y_pred"], x["y"]))

    def _init_model(self):
        if self.hparams.model_name == "sed_ccnn14att":
            from models.sed_models import PANNsCNN14Att
            self.model = PANNsCNN14Att(**self.hparams.model_config)
            if self.hparams.pretrained_path is not None:
                weights = torch.load(self.hparams.pretrained_path, map_location="cpu")["model"]
                weight_keys = weights.copy().keys()
                for key in weight_keys:
                    if "att_block" in key:
                        del weights[key]
                self.model.load_state_dict(weights, strict=False)
        elif self.hparams.model_name == "sed_dense121att":
            from models.sed_models import PANNsDense121Att
            self.model = PANNsDense121Att(**self.hparams.model_config)
        elif self.hparams.model_name == "sed_dense161att":
            from models.sed_models import PANNsDense161Att
            self.model = PANNsDense161Att(**self.hparams.model_config)
        elif self.hparams.model_name == "sed_dense169att":
            from models.sed_models import PANNsDense169Att
            self.model = PANNsDense169Att(**self.hparams.model_config)
        elif self.hparams.model_name == "sed_dense201att":
            from models.sed_models import PANNsDense201Att
            self.model = PANNsDense201Att(**self.hparams.model_config)
           
    def _init_augmentation(self):
        if self.hparams.aug_name == "default":
            from augmentations.sed_default_augment import get_transforms
            self.tfms = get_transforms(bckgrd_aug_dir=self.hparams.bckgrd_aug_dir)
        elif self.hparams.aug_name == "secondary_default":
            from augmentations.sed_background_augment import get_transforms
            self.tfms = get_transforms(bckgrd_aug_dir=self.hparams.bckgrd_aug_dir, secondary_bckgrd_aug_dir=self.hparams.secondary_bckgrd_aug_dir)

        if self.hparams.apply_mixup:
            from augmentations.mixup import Mixup, do_mixup
            self.mixup_augmenter = Mixup(mixup_alpha=1.)
        else:
            self.mixup_augmenter = None
        
    def _init_train_datalader(self):
        from dataloaders.sed_dataset import SedDataset
        self.train_ds = SedDataset(**self.hparams.train_ds_params, transform=self.tfms["train"])

    def _init_valid_dataloader(self):
        from dataloaders.sed_dataset import SedDataset
        self.valid_ds = SedDataset(**self.hparams.valid_ds_params, transform=self.tfms["valid"])

    def _init_test_dataloader(self):
        from dataloaders.sed_dataset import SedDataset
        self.test_ds = SedDataset(**self.hparams.test_ds_params, transform=self.tfms["valid"])
            
        
