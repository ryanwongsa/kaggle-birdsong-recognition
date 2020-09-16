from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from logger.base.base_logger import BaseLogger
from logger.base.utils import *
from logger.neptune.neptune_utils import *
from ignite.contrib.handlers.neptune_logger import *

import numpy as np
import os

class MyNeptuneLogger(BaseLogger):
    
    def __init__(self, log_every=5, **kwargs):
        self.writer = NeptuneLogger(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                           project_name=kwargs["project_name"],
                           name=kwargs["name"],
                           params=kwargs["params"],
                           tags=kwargs["tags"])
        super().__init__(log_every=log_every)

    def _add_train_events(self, model = None, optimizer=None, scheduler=None, metrics={}):
        # self.writer.attach(self.trainer,
        #           log_handler=WeightsScalarHandler(model),
        #           event_name=Events.ITERATION_COMPLETED(every=100))
        # self.writer.attach(self.trainer,
        #           log_handler=GradsScalarHandler(model),
        #           event_name=Events.ITERATION_COMPLETED(every=100))
                    
        iteration_events = [
            training_iteration(self.writer),
            lr_iteration(optimizer, self.writer)
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
        self.writer.experiment.stop()