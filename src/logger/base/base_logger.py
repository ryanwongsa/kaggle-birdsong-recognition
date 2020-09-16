from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from logger.base.utils import *

class BaseLogger(object):
    
    def __init__(self, log_every=10):
        self.log_every = log_every
        self.trainer = None
        self.evaluator = None
    
    def _init_logger(self, trainer, evaluator):
        self.trainer = trainer
        self.evaluator = evaluator
        
    def _add_train_handlers(self, iteration_events = [], completion_events = []):
        add_iteration_handlers(self.trainer, iteration_events, self.log_every)
        add_completion_handlers(self.trainer, completion_events)
    
    def _add_evaluation_handlers(self, iteration_events = [], completion_events = []):
        add_iteration_handlers(self.evaluator, iteration_events, self.log_every)
        add_completion_handlers(self.evaluator, completion_events)
        
    def _add_train_events(self, model = None, optimizer=None, scheduler=None, metrics={}):
        raise NotImplementedError
        
    def _add_eval_events(self, model = None, optimizer=None, scheduler=None, metrics={}):
        raise NotImplementedError
        
    def _add_custom_train_iteration_handler(self, iteration_event, log_every):
        add_iteration_handlers(self.trainer, [iteration_event], log_every)
        
    def _add_custom_eval_iteration_handler(self, iteration_event, log_every):
        add_iteration_handlers(self.evaluator, [iteration_event], log_every)