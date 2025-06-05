from .utils import load_config
import torch
import argparse

def load_parameter():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='hyper-parameter file name')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    args = parser.parse_args()
    
    config_name = args.config
    config_path = f'kg_config/{config_name}.yaml'
    # load config 
    config = load_config(config_path)
    
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    
    # check cuda
    if torch.cuda.is_available():
        print('cuda')
        args_cuda = True
    else:
        print('cpu')
        args_cuda = False
    config['args_cuda'] = args_cuda
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    config['iter_num'] = 1 # since both Aminer and DBLP is trained on the full graph without batch training

    config['num_levels'] = config['layer_num']
    if len(config['task_name']) <= 1:
        raise ValueError("The number of tasks must be greater than 1 in multi-task learning model!")

    
    if 'loss_weight' not in config:
        config['loss_weight'] = [1.]*len(config['task_name'])
        

    return config