import torch
import requests
import os

try:
    slack_url = os.environ["SLACK_URL"]
except:
    slack_url = None
    
def lr_iteration(optimizer,writer):
    def print_lr(engine):
        for i, pg in enumerate(optimizer.param_groups):
            print(f'train/lr{i}', engine.state.iteration - 1, optimizer.param_groups[i]['lr']) 
    return print_lr

def training_iteration(writer):
    def print_training_iteration(engine):
        iteration = engine.state.iteration
        print('train/loss', iteration, engine.state.output['loss'].item()) 
        if "dict_loss" in engine.state.output:
            for key, value in engine.state.output['dict_loss'].items():
                print(f"train/{key}", iteration, value.item())
    return print_training_iteration

def train_metrics_completion(writer):
    def print_train_metrics_completion(engine):
        metrics = engine.state.metrics
        for key, value in metrics.items():
            print(f"train/{key}",  engine.state.epoch, value)
        if slack_url is not None:
            try:
                requests.post(slack_url, json={"text": f"[TRAIN] {engine.state.epoch}: {metrics['train_avg_loss']}"})
            except:
                pass
    return print_train_metrics_completion

def validation_metrics_completion(trainer, writer):
    def print_metrics(engine):
        metrics = engine.state.metrics
        for key, value in metrics.items():
            print(f"valid/{key}", trainer.state.epoch, value)
        if slack_url is not None:
            # requests.post(slack_url, json={"text": f"[VALID] {trainer.state.epoch}: {metrics['valid_avg_loss']}"})
            for key, value in metrics.items():
                try:
                    requests.post(slack_url, json={"text": f"[VALID-{key}] {trainer.state.epoch} : {value}"})
                except:
                    pass
    return print_metrics

