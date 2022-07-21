import json
import os
import random
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb


def fix_random_seed_as(random_seed, use_cudnn=False):
    if random_seed == -1:
        random_seed = np.random.randint(100000)
        print("RANDOM SEED: {}".format(random_seed))

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    if use_cudnn:
        cudnn.benchmark = True
        cudnn.enabled = True
    else:
        cudnn.deterministic = True
        cudnn.benchmark = False


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def create_experiment_export_folder(experiment_dir, experiment_description):
    print(os.path.abspath(experiment_dir))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    print(os.path.abspath(experiment_path))
    os.mkdir(experiment_path)
    print("folder created: " + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def export_config_as_json(config, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=2)


def setup_experiment(config):
    config.lr = config.lr / 16 * config.batch_size
    fix_random_seed_as(config['random_seed'], use_cudnn=config.use_cudnn)
    export_root = create_experiment_export_folder(config['experiment_dir'], config['experiment_name'])
    config.export_root = export_root
    setup_wandb(config)
    return export_root, config


def setup_experiment_without_wandb(config):
    config.lr = config.lr / 16 * config.batch_size
    fix_random_seed_as(config['random_seed'], use_cudnn=config.use_cudnn)
    export_root = create_experiment_export_folder(config['experiment_dir'], config['experiment_name'])
    config.export_root = export_root

    return export_root, config


def setup_wandb(config):
    os.environ['WANDB_DISABLE_CODE'] = "true"
    os.environ['WANDB_SILENT'] = "true"
    if config.debug_mode:
        os.environ["WANDB_MODE"] = "dryrun"
    model_name = config.get('model', None)
    method_name = config.get('method', None)
    tags = [model_name, method_name]
    tags = [val for val in tags if val is not None]
    save_dir = config.get('export_root', '/home/dongwan/wandb')
    wandb.init(config=config, name=config.experiment_name, project=config.project_name, entity=config.wandb_entity,
               tags=tags, dir=save_dir)
    print("Wandb Setup complete")


def create_exp_name(model_folder):
    import yaml
    wandb_folder_path = os.path.join(model_folder, 'wandb')
    run_folder_path = sorted(os.listdir(wandb_folder_path))[-1]
    config_path = os.path.join(wandb_folder_path, run_folder_path, 'config.yaml')
    with open(config_path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    configs = data['content']['value']
    exp_name = configs['experiment_name']
    return exp_name
