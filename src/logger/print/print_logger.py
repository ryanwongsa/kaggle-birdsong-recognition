from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from logger.base.base_logger import BaseLogger
from logger.base.utils import *
from logger.print.print_utils import *
from ignite.contrib.handlers.neptune_logger import *

import numpy as np
import os

class PrintLogger(BaseLogger):
    
    def __init__(self, log_every=5, **kwargs):
        self.writer = None
        super().__init__(log_every=log_every)

    def _add_train_events(self,  model = None, optimizer=None, scheduler=None, metrics={}):
        
        iteration_events = [
            training_iteration(self.writer),
            # lr_iteration(optimizer, self.writer)
        ]

        completion_events = [
            train_metrics_completion(self.writer)
        ]
        
        self._add_train_handlers(
            **{
                "iteration_events": iteration_events,
                "completion_events": completion_events
            }
        )
            
    def _add_eval_events(self,  model = None, optimizer=None, scheduler=None, metrics={}):
        iteration_events = []
        
        completion_events = [
            validation_metrics_completion(self.trainer, self.writer),
        ]
        
        self._add_evaluation_handlers(
            **{
                "iteration_events": iteration_events,
                "completion_events": completion_events
            }
        )
        
    def _end_of_training(self):
        return