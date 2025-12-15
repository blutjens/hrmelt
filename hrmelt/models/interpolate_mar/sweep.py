import argparse
import optuna
from predict import predict
import yaml
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Create persistence predictions on validation set')
    parser.add_argument('--cfg_path', type=str, default='runs/interpolate_mar/sample/config/config.yaml',
                        help='Path to config yaml')
    parser.add_argument('--sweep_path', type=str, default='runs/interpolate_mar/sample/config/sweep.yaml',
                        help='Path to config yaml')
    return parser.parse_args()

# Define your objective function
def objective(trial):

    args = get_args()
    cfg = yaml.safe_load(open(args.cfg_path, 'r'))
    sweep_cfg = yaml.safe_load(open(args.sweep_path, 'r'))

    # Init cpu or gpu
    if cfg['use_gpu']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Using device {device}')

    # Define the parameter space to search
    blur = trial.suggest_categorical('blur', sweep_cfg['blur'])
    kernel_size = trial.suggest_int('kernel_size', sweep_cfg['kernel_size_min'], sweep_cfg['kernel_size_max'], step=2)
    sigma = trial.suggest_int('sigma', sweep_cfg['sigma_min'], sweep_cfg['sigma_max'], step=2)
    gamma = trial.suggest_float('gamma', sweep_cfg['gamma_min'], sweep_cfg['gamma_max'])
    brightness_factor = trial.suggest_float('brightness_factor', sweep_cfg['brightness_factor_min'], sweep_cfg['brightness_factor_max'])

    cfg['blur'] = blur
    cfg['kernel_size'] = kernel_size
    cfg['sigma'] = sigma
    cfg['gamma'] = gamma
    cfg['brightness_factor'] = brightness_factor

    # Call your function with the sampled parameters
    loss = predict(cfg, split='train', device=device)
    # Return the metric to be minimized
    return loss

if __name__ == '__main__':

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')  # 'minimize' because we want to minimize the objective
    study.optimize(objective, n_trials=200)  # You can adjust the number of trials

    # Get the best parameters and result
    best_params = study.best_params
    best_result_metric = study.best_value

    print("Best Parameters:", best_params)
    print("Best Result Metric:", best_result_metric)
