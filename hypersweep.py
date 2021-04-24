import argparse
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import os

import mlflow

from auto_train import *


def objective(trial, args):
    
    # fixed hyperparams
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, str(trial.number))
    args.name = str(trial.number)
    
    # hyperparams to be optimized
    # args.model =     # model switching will follow later
    args.lr = trial.suggest_discrete_uniform('lr', 0.001, 0.015, 0.001)
    args.lr_gamma = trial.suggest_loguniform('lr_gamma', 0.005, 0.5)
    args.lr_step_size = trial.suggest_discrete_uniform('lr_step_size', 3, 24, 1)
    args.momentum = trial.suggest_uniform('momentum', 0, 0.99)
    args.weight_decay = trial.suggest_discrete_uniform('weight_decay', 0.0001, 0.0005, 0.0001)
    
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.name):
        metric = main(args, trial)
    
    return metric

if __name__ == '__main__':
    
    args = parse_args()
    storage='sqlite:///demo.db'
    study = optuna.load_study(study_name=args.experiment_name,
                              storage=storage,
                              sampler=TPESampler(n_startup_trials=8),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                                                 n_warmup_steps=20,
                                                                 interval_steps=20))
    study.optimize(lambda trial: objective(trial, args), n_trials=20, catch=(RuntimeError,))