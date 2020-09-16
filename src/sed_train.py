from argparse import ArgumentParser, Namespace
from engine.sed_engine import SedEngine
import importlib
import torch
import ignite.distributed as idist
torch.backends.cudnn.benchmark = True

def run(local_rank, config):
    pe = SedEngine(local_rank, config)
    pe.train(config.run_params)

def main(hyperparams): 
    with idist.Parallel(**hyperparams.dist_params) as parallel:
        parallel.run(run, hyperparams)

if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    
    parser.add_argument('--config', type=str)
    
    params = parser.parse_args()
    
    module = importlib.import_module(params.config, package=None)
    hyperparams = module.Parameters()
    
    main(hyperparams)