from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

def add_iteration_handlers(engine, method_events = [], log_every=5):
    for method_event in method_events:
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=log_every), method_event)

def add_completion_handlers(engine, method_events = []):
    for method_event in method_events:
        engine.add_event_handler(Events.EPOCH_COMPLETED, method_event)
        
