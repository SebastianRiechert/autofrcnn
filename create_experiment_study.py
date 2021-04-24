import mlflow
import optuna
from optuna.samplers import TPESampler

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name for study in optuna and experiment in mlflow')
    parser.add_argument('--db', default='sqlite:///demo.db' , help='connection string to where optuna stores studies')
    
    args = parser.parse_args()

    storage=args.db
    mlflow.set_experiment(args.name)
    study = optuna.create_study(study_name=args.name,
                                direction="maximize",
                                storage=storage,
                                sampler=TPESampler(n_startup_trials=8),
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                                                   n_warmup_steps=20,
                                                                   interval_steps=20))