from argparse import ArgumentParser, Namespace
from engine.main_engine_v2 import MainEngineV2
import importlib
import torch
import ignite.distributed as idist

def run(local_rank, config):
    pe = MainEngineV2(local_rank, config)
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