
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict
import math
import os
import io
import time
import copy
import sys
import json
import logging
import random
import numpy as np
from tqdm import tqdm
from datetime import date, datetime, timedelta
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from kg_lib.args import load_parameter
from pub_data.dgl_generator import load_dgl_data
from kg_lib.utils import mkdir
from kg_lib.logging_util import init_logger
from kg_lib.evaluation_util import evaluation
from kg_model.struchis import StrucHIS

# Intial seed
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



BCE_loss = torch.nn.BCELoss()   

def train(config, dataloader, logging, result_output_dir, multi_train, tt):  
    
    args_cuda = config['args_cuda']
    evaluation_func = evaluation(config['dataset'])
        
    # model
    model = StrucHIS(config)
    if args_cuda:
        model.cuda()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr = config['learning_rate'],
                                weight_decay = config['weight_decay'])
    
    # Initialize best metric result
    if config['save_metric'] == 'loss':
        best_roc_auc = [999] * len(config['task_name'])
    else:
        best_roc_auc = [0] * len(config['task_name'])
    early_stop_count = 0
    print_result_best = ''

    metric_list_dict_single_ep = defaultdict(lambda: defaultdict(dict))
    metric_list_dict_single_ep_best = None
    # metric_record
    metric_list_dict = {}
    metric_list_dict['train'] = defaultdict(lambda: defaultdict(list))
    metric_list_dict['val'] = defaultdict(lambda: defaultdict(list))
    metric_list_dict['test'] = defaultdict(lambda: defaultdict(list))



    for epoch in range(config['train_epoch']):
        
        model.train()
    
        train_per_epoch(config, model, dataloader, optimizer, multi_train)
        

        if epoch%config['eval_inter_ep'] == 0:
            print('Epoch:', epoch)
            print_result = ''
            for data_type in ['train', 'val', 'test']:
                result = evaluation_func(model, dataloader, data_type, config)
                print_result += f'\n{data_type} - '
                for label_name in result:
                    for metric in result[label_name]:
                        metric_value = result[label_name][metric]
                        metric_list_dict[data_type][label_name][metric].append(metric_value)
                        metric_list_dict_single_ep[data_type][label_name][metric] = metric_value
                        if (type(metric_value)!=dict) and (type(metric_value)!=list):
                            print_result += f'{label_name}_{metric}:{metric_value:.4f}, '
                        else:
                            print_result += f'{label_name}_{metric}:{metric_value}, '
                    print_result += '\n'


            logging.info(f'Epoch:{epoch}')
            logging.info(print_result)
            if not multi_train:
                print(print_result)


            # save model
            val_roc_auc = [metric_list_dict['val'][label_name][config['save_metric']][-1] for label_name in config['task_name']]
            if config['save_metric'] == 'loss':
                dif = np.array(best_roc_auc)-np.array(val_roc_auc)
            else:
                dif = np.array(val_roc_auc)-np.array(best_roc_auc)
            if np.all(dif>=0):
                early_stop_count = 0
                best_roc_auc = val_roc_auc
                print_result_best = print_result
                metric_list_dict_single_ep_best = copy.deepcopy(metric_list_dict_single_ep)
                model_save_name = 'model_parameter_best_roc_auc_'
                for ind, label_name in enumerate(config['task_name']):
                    model_save_name += f'_{label_name}-{best_roc_auc[ind]:.4f}'
                if multi_train:
                    torch.save(model.state_dict(), os.path.join(result_output_dir, f'best_model_{tt}.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(result_output_dir, model_save_name+'.pt'))
            else:
                early_stop_count = early_stop_count + 1
                if not multi_train:
                    print("Early Stop Count:", early_stop_count)
                logging.info(f"Early Stop Count:{early_stop_count}")

                if early_stop_count >= config['early_stop']:
                    return print_result_best, metric_list_dict_single_ep_best, metric_list_dict
                
    return print_result_best, metric_list_dict_single_ep_best, metric_list_dict



def train_per_epoch(config, model, dataloader, optimizer, multi_train):
    args_cuda = config['args_cuda']
        
    hg_dgl = dataloader.hg_dgl

    if multi_train:
        pbar = range(config['iter_num'])
    else:
        pbar = tqdm(range(config['iter_num']))

    for sample_index in pbar:

        loss = 0

        h_output = model(hg_dgl)
        result_pbar = {}

        for ind, task_name in enumerate(h_output):
            output = h_output[task_name]

            target_node_name = config['task_to_node'][task_name]
            mask = hg_dgl.nodes[target_node_name].data['train_label_mask']
            tmp_true_label = hg_dgl.nodes[target_node_name].data['train_label'][mask,:]

            output = output[mask]
            loss_i = BCE_loss(output, tmp_true_label)
            loss += config['loss_weight'][ind]*loss_i


            if args_cuda:
                source_data_label_np = tmp_true_label.data.cpu().numpy()
                h_output_squeeze_np = output.data.cpu().numpy()
            else:
                source_data_label_np = tmp_true_label.data.numpy() 
                h_output_squeeze_np = output.data.numpy()

            task_type = config['task_type'][task_name]
            if task_type == 'single-label':
                h_output_squeeze_np = h_output_squeeze_np.argmax(axis=1)
                source_data_label_np = source_data_label_np.argmax(axis=1)
            elif task_type == 'multi-label':
                h_output_squeeze_np = (h_output_squeeze_np>0.5).astype(int)
            else:
                raise NameError(f'task type {task_type} does not exist!')

            micro = f1_score(source_data_label_np, h_output_squeeze_np, average='micro')
            macro = f1_score(source_data_label_np, h_output_squeeze_np, average='macro')

            result_pbar[task_name + '_loss'] = loss_i.item()
            result_pbar[task_name + '_micro'] = micro
            result_pbar[task_name + '_macro'] = macro

        if not multi_train:
            pbar.set_postfix(result_pbar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__=='__main__':      
    ### Load Hyper-parameters
    config = load_parameter()
    print(config)   
    setup_seed(config['seed'])


    ### Data Preprocessing
    print('Loading data......')
    dataloader = load_dgl_data(config)
    config['edge_dict'] = dataloader.edge_dict
    config['node_dict'] = dataloader.node_dict
    config['node_type_to_feature_len_dict'] = dataloader.node_type_to_feature_len_dict
    config['edge_type_each_node'] = dataloader.edge_type_each_node

       
    ### Create ouput path
    localtime = time.strftime("%m-%d-%H:%M:%S", time.localtime())
    result_output_dir = "Result/" + config['dataset'] + "/" + localtime
    mkdir('Result')
    mkdir("Result/" +  config['dataset'])
    mkdir(result_output_dir)
    print('Output path:', result_output_dir)
    # logging
    log_filename = os.path.join(result_output_dir, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    logging.info(config)
    # best result
    best_result_save_file_name = os.path.join(result_output_dir, 'best_result.json')

    

    ### Running
    rerun_num = config['rerun_num']
    multi_train = True
    metric_list_dict_best = []
    for tt in range(rerun_num):
        print_result_best, metric_list_dict_single_ep, metric_list_dict = train(config, dataloader, logging, result_output_dir, multi_train, tt)
        logging.info("\n\n\n\n-----------------------------------------------------------------\n\n\n\n")
        logging.info("\n----------------------------Result-------------------------------------")
        logging.info(print_result_best)
        metric_list_dict_best.append(copy.deepcopy(metric_list_dict_single_ep))
        with open(best_result_save_file_name, 'w') as outfile:
            json.dump(metric_list_dict_best, outfile)
        with open(os.path.join(result_output_dir, f'metric_record_{tt}.json'), 'w') as outfile:
            json.dump(metric_list_dict, outfile)    

        logging.info("----------------------------------------------------------------")



    ### Show multi-times results
    with open(result_output_dir + '/best_result.json') as f:
        metric_list_dict_all = json.load(f)
    result_mean = defaultdict(lambda: defaultdict(dict))
    len_result = len(metric_list_dict_all)
    for result in metric_list_dict_all:
        for data_type in result:
            for label in result[data_type]:
                for metric in result[data_type][label]:
                    if metric in result_mean[data_type][label]:
                        result_mean[data_type][label][metric].append(result[data_type][label][metric])
                    else:
                        result_mean[data_type][label][metric] = [result[data_type][label][metric]]
    for data_type in result:
        for label in result[data_type]:
            for metric in result[data_type][label]:
                aa = result_mean[data_type][label][metric] 
                if type(aa[0])!=dict:
                    aa_mean = np.mean(aa)*100
                    aa_std = np.std(aa)*100
                    print(f'{data_type}-{label}-{metric}-{aa_mean:.2f}+{aa_std:.2f}')
                    logging.info(f'{data_type}-{label}-{metric}-{aa_mean:.2f}+{aa_std:.2f}')







