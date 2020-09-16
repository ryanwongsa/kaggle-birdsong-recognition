import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import math

try:
    from apex import amp
    print("Found NVIDIA Apex, using mixed precision")
    print("NOT USING APEX BECAUSE CANNOT CREATE MEL SPEC, REQUIRES TUNING, PROBLEMS HAVE OCCURRED")
    USE_AMP = False
except:
    print("NVIDIA Apex not found, not using mixed precision")
    USE_AMP = False

from tqdm.auto import tqdm
import os
import cProfile
import ignite
from ignite.engine import Events, Engine
from ignite.handlers import Checkpoint
from ignite.utils import manual_seed, setup_logger
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import FastaiLRFinder

try:
    XLA_USE_BF16 = os.getenv(XLA_USE_BF16)
except:
    XLA_USE_BF16 = 0
EPSILON = 1e-6
EPSILON_FP16 = 1e-5

class BaseEngine(object):
    def __init__(self, local_rank, hparams):
        self.local_rank = local_rank
        self.hparams = hparams

        self._init_score_function()
        self._init_augmentation()

        self._init_train_datalader()
        self._init_valid_dataloader()
        self._init_test_dataloader()

        self._init_model()
        self._init_criterion_function()
        self._init_metrics()
        
        self.setup()
    
    def _init_distribution(self):
        self.rank = idist.get_rank()
        manual_seed(42+ self.rank)
        self.device = idist.device()

        if self.train_ds:
            if self.train_ds.sampler is not None:
                sampler = self.train_ds.sampler(self.train_ds, self.train_ds.get_label)
                isShuffle=False
            else:
                sampler = None
                isShuffle=True
            self.train_loader = idist.auto_dataloader(
                self.train_ds, batch_size=self.hparams.train_bs, num_workers=self.hparams.train_num_workers, shuffle=isShuffle, drop_last=True, sampler=sampler, **self.train_ds.additional_loader_params
            )
        
        if self.valid_ds:
            self.valid_loader = idist.auto_dataloader(
                self.valid_ds, batch_size=self.hparams.valid_bs, num_workers=self.hparams.valid_num_workers, shuffle=False, drop_last=False, **self.valid_ds.additional_loader_params
            )

        if self.test_ds:
            self.test_loader = idist.auto_dataloader(
                self.test_ds, batch_size=self.hparams.valid_bs, num_workers=self.hparams.valid_num_workers, shuffle=False, drop_last=False, **self.test_ds.additional_loader_params
            )
        
        if USE_AMP:
            self._init_optimizer()
            self.model = idist.auto_model(self.model)
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        else:
            self.model = idist.auto_model(self.model)
        
        if not USE_AMP:
            self._init_optimizer()
        
        self.optimizer = idist.auto_optim(self.optimizer)
        
        self._init_scheduler()
        
        self.criterion = self.criterion.to(self.device)

    def log_basic_info(self, logger):
        logger.info("- PyTorch version: {}".format(torch.__version__))
        logger.info("- Ignite version: {}".format(ignite.__version__))
        if idist.get_world_size() > 1:
            logger.info("\nDistributed setting:")
            logger.info("\tbackend: {}".format(idist.backend()))
            logger.info("\tworld size: {}".format(idist.get_world_size()))
            logger.info("\n")

    def load_trainer_from_checkpoint(self):
        if self.hparams.checkpoint_dir is not None:
            if not self.hparams.load_model_only:
                objects_to_checkpoint = {
                    "trainer": self.trainer,
                    "model": self.model, 
                    "optimizer": self.optimizer,
                    "scheduler": self.scheduler
                }
                if USE_AMP:
                    objects_to_checkpoint["amp"] = amp
            else:
                objects_to_checkpoint = {"model": self.model}
            objects_to_checkpoint = {k: v for k, v in objects_to_checkpoint.items() if v is not None}
            checkpoint = torch.load(self.hparams.checkpoint_dir, map_location="cpu")
            Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

    def setup_checkpoint_saver(self, to_save):
        if self.hparams.checkpoint_params is not None:
            from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

            handler = Checkpoint(to_save, DiskSaver(self.hparams.checkpoint_params["save_dir"], require_empty=False), n_saved=self.hparams.checkpoint_params["n_saved"],
                                filename_prefix=self.hparams.checkpoint_params["prefix_name"], score_function=self.score_function, score_name="score", 
                                global_step_transform=global_step_from_engine(self.trainer))

            self.evaluator.add_event_handler(Events.COMPLETED, handler)

    def attach_metrics(self, engine, metrics):
        for name, metric in metrics.items():
            metric.attach(engine, name)

    def setup(self):
        self._init_distribution()

        self.trainer = Engine(self.train_step)
        self.trainer.logger = setup_logger(name="trainer", distributed_rank=self.local_rank)
        self.log_basic_info(self.trainer.logger)

        self.load_trainer_from_checkpoint()

        if self.scheduler:
            self.scheduler_event = self.trainer.add_event_handler(Events.ITERATION_STARTED, self.scheduler)
        else:
            self.scheduler_event = None
        self.attach_metrics(self.trainer, self.train_metrics)
        

        if idist.get_world_size() >1:
            def set_epoch(engine):
                self.train_loader.sampler.set_epoch(engine.state.epoch)

            self.trainer.add_event_handler(Events.EPOCH_STARTED, set_epoch)


        common.setup_common_training_handlers(
            self.trainer,
            train_sampler=self.train_loader.sampler,
            to_save=None,
            save_every_iters=0,
            output_path= None,
            lr_scheduler= None,
            output_names= None,
            with_pbars=self.hparams.add_pbar,
            clear_cuda_cache=True,
            stop_on_nan=False
        )
        
        self.evaluator = Engine(self.eval_step)
        self.evaluator.logger = setup_logger("evaluator", distributed_rank=self.local_rank)
        if self.hparams.add_pbar:
            ProgressBar(persist=False).attach(self.evaluator)

        def complete_clear(engine):
            engine.state.batch = None
            engine.state.output = None
            import gc
            gc.collect()
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, complete_clear)

        self.validation_handler_event = self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.hparams.eval_every), self.validate(self.valid_loader))
        self.evaluator.add_event_handler(Events.EPOCH_COMPLETED, complete_clear)

        train_handler_params = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler
        }

        eval_handler_params = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler
        }

        to_save = {
                "model": self.model,
                "trainer": self.trainer,
                "optimizer": self.optimizer
            }
        if self.scheduler is not None:
            to_save["scheduler"] = self.scheduler
        if USE_AMP:
            to_save["amp"] = amp
        self.attach_metrics(self.evaluator, self.validation_metrics)
        self.setup_checkpoint_saver(to_save)
        
        if self.rank == 0:
            self._init_logger()
            if self.logger:
                self.logger._init_logger(self.trainer, self.evaluator)
                self.logger._add_train_events(**train_handler_params)
                self.logger._add_eval_events(**eval_handler_params)
    
    def train(self, run_params):
        try:
            self.trainer.run(self.train_loader,**run_params)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
        if self.rank==0 and self.logger:
            self.logger._end_of_training()
            
    def validate(self, dl):
        def validate_run(engine):
            self.evaluator.run(dl, epoch_length=self.hparams.val_length)
        return validate_run
            
    def evaluate(self, dl):
        self.tester = Engine(self.test_step)
        self.attach_metrics(self.tester, self.test_metrics)
        if self.hparams.add_pbar:
            ProgressBar(persist=False).attach(self.tester)
        self.tester.run(dl, epoch_length=self.hparams.val_length)
    
    def train_step(self, engine, batch):
        engine.state.output = None
        self.model.train()
        x, y = self.prepare_batch(batch, mode = 'train')
        y_pred = self.model(x)
        loss, dict_loss = self.loss_fn(y_pred, y)
        self.loss_backpass(loss)
        
        if engine.state.iteration % self.hparams.accumulation_steps == 0:
            self.clip_gradients()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return self.output_transform(x, y, y_pred, loss, dict_loss, mode = 'train')
    
    def loss_backpass(self, loss):
        if USE_AMP:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    
    def eval_step(self, engine, batch):
        engine.state.output = None
        self.model.eval()
        with torch.no_grad():
            x, y = self.prepare_batch(batch, mode = 'valid')
            y_pred = self.model(x)
            loss, dict_loss = self.loss_fn(y_pred, y)
            return self.output_transform(x, y, y_pred, loss, dict_loss, mode = 'valid')
    
    
    def test_step(self, engine, batch):
        engine.state.output = None
        self.model.eval()
        with torch.no_grad():
            x, y = self.prepare_batch(batch, mode = 'test')
            y_pred = self.model(x)
            return self.output_transform(x, y, y_pred, mode='test')
    
    def prepare_batch(self, batch, model='valid'):
        return batch
    
    def output_transform(self, x, y, y_pred, loss=None):
        return x, y, y_pred, loss
    
    def _init_scheduler(self):
        self.scheduler = None
    
    def get_batch(self):
        return next(iter(self.train_loader))
    
    def loss_fn(self, y_pred, y):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError
        
    def _init_optimizer(self):
        raise NotImplementedError

    def _init_criterion_function(self):
        raise NotImplementedError
        
    def _init_logger(self):
        raise NotImplementedError
        
    def _init_metrics(self):
        raise NotImplementedError
    
    def _init_train_datalader(self):
        self.train_ds = None
        
    def _init_valid_dataloader(self):
        self.valid_ds = None

    def _init_test_dataloader(self):
        self.test_ds = None
    
    def _init_augmentation(self):
        self.tfms = None
    
    def _init_score_function(self):
        def get_score_function(metric_name, factor=1):
            def score_function(engine):
                return factor*engine.state.metrics[metric_name]
            return score_function
        
        self.score_function = get_score_function(
            self.hparams.track_metric, self.hparams.metric_factor)
    
    def clip_gradients(self):
        # Source https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/trainer/training_tricks.py
        if self.hparams.gradient_clip_val <= 0:
            return
        parameters = self.model.parameters()
        max_norm = float(self.hparams.gradient_clip_val)
        norm_type = float(2.0)
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            device = parameters[0].device
            out = torch.empty(len(parameters), device=device)
            for i, p in enumerate(parameters):
                torch.norm(p.grad.data.to(device), norm_type, out=out[i])
            total_norm = torch.norm(out, norm_type)

        eps = EPSILON_FP16 if USE_AMP or XLA_USE_BF16==1 else EPSILON
        clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        for p in parameters:
            p.grad.data.mul_(clip_coef.to(p.grad.data.device))


    def lr_finder(self, min_lr=0.00003, max_lr=10.0, num_iter=None):
        trainer = self.trainer
        self.validation_handler_event.remove()
        if self.scheduler_event is not None:
            self.scheduler_event.remove()
        model = self.model
        optimizer = self.optimizer
        dataloader = self.train_loader
        if num_iter is None:
            num_iter = len(dataloader)-1

        lr_finder = FastaiLRFinder()
        to_save = {"model": model, "optimizer": optimizer}

        for param_group in optimizer.param_groups:
            param_group['lr'] = min_lr

        def output_transform(x):
            return x["loss"]

        with lr_finder.attach(trainer, num_iter=num_iter, end_lr=max_lr, to_save=to_save, output_transform=output_transform) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(dataloader)
        return lr_finder