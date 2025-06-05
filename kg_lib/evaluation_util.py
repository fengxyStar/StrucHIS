import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve, roc_curve, f1_score
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

import time

BCE_loss = torch.nn.BCELoss()


def evaluation(dataset):
    if dataset == 'Aminer' or dataset == 'DBLP':
        return evaluate_func
    elif dataset == 'Logistic':
        return evaluate_Logistic
    else:
        raise NameError(f"Evaluation does not support the dataset {dataset}. Please try DBLP/Aminer/Logistic.")

    

def evaluate_func(model, dataloader, data_type, config):
    model.eval()
    result = {}
    hg_dgl = dataloader.hg_dgl
    with torch.no_grad():   
        h_output = model(hg_dgl)
        loss = 0
        for target_node_name in h_output:
            h_output_squeeze = h_output[target_node_name]
            mask = hg_dgl.nodes[target_node_name].data[f'{data_type}_label_mask']
            source_data_label = hg_dgl.nodes[target_node_name].data[f'{data_type}_label'][mask]
            
            h_output_squeeze = h_output_squeeze[mask]
            
            loss_i = BCE_loss(h_output_squeeze, source_data_label)

            loss = loss +  loss_i
    
            if config['args_cuda']:
                source_data_label_np = source_data_label.data.cpu().numpy()
                h_output_squeeze_np = h_output_squeeze.data.cpu().numpy()
            else:
                source_data_label_np = source_data_label.data.numpy()
                h_output_squeeze_np = h_output_squeeze.data.numpy()
            
            task_type = config['task_type'][target_node_name]
            if  task_type == 'single-label':
                h_output_squeeze_np = h_output_squeeze_np.argmax(axis=1)
                source_data_label_np = source_data_label_np.argmax(axis=1)
            elif task_type == 'multi-label':
                h_output_squeeze_np = (h_output_squeeze_np>0.5).astype(int)
            else:
                raise NameError(f'task type {task_type} does not exist!')
            micro = f1_score(source_data_label_np, h_output_squeeze_np, average='micro')
            macro = f1_score(source_data_label_np, h_output_squeeze_np, average='macro')
            
            
            result[target_node_name] = {'loss':loss_i.item(), 'micro': micro, 'macro': macro}
    
    return result



def evaluate_Logistic(model, Logistic_dataloader, data_type, config):
    model.eval()
    label_name = config['task_name'][0]
    Target_Node_Type = config['Target_Node_Type']
    tmp_time_range_list = Logistic_dataloader.KG_time_list[data_type]
    with torch.no_grad():    
        h_output_squeeze_list = defaultdict(list)
        source_data_label_list = defaultdict(list)
        for tmp_time_range in tmp_time_range_list:
            time_range_str = ('Time_Range:' + str(tmp_time_range))
            source_Data_Dict = Logistic_dataloader.time_range_to_Processed_Subgraph_Data_dict[time_range_str]
            for sample_start in tqdm(range(0, source_Data_Dict[label_name].shape[0], config['eval_sample_size'][0])):
                sample_end = sample_start + config['eval_sample_size'][0]
                if sample_end > source_Data_Dict[label_name].shape[0]:
                    sample_end = source_Data_Dict[label_name].shape[0]
                
                sampled_hg = Logistic_dataloader.sample_sub_graph_with_index(source_Data_Dict, np.arange(sample_start,sample_end))
                h_output = model(sampled_hg)
                
                for label_name in config['task_name']:   
                    h_output_squeeze = torch.squeeze(h_output[label_name])

                    tmp_true_label = sampled_hg.nodes[Target_Node_Type].data[label_name]
                    tmp_true_label_index = (tmp_true_label != -1).nonzero()

                    h_output_squeeze_list[label_name].append(h_output_squeeze[tmp_true_label_index])
                    source_data_label_list[label_name].append(tmp_true_label[tmp_true_label_index])
    
    result = {}
    for label_name in config['task_name']:  
        h_output_squeeze = torch.cat(h_output_squeeze_list[label_name])
        source_data_label = torch.cat(source_data_label_list[label_name])

        loss = BCE_loss(h_output_squeeze, source_data_label)
        
        if config['args_cuda']:
            source_data_label_np = source_data_label.data.cpu().numpy()
            h_output_squeeze_np = h_output_squeeze.data.cpu().numpy()
        else:
            source_data_label_np = source_data_label.data.numpy()
            h_output_squeeze_np = h_output_squeeze.data.numpy()

        fpr, tpr, thresholds = roc_curve(source_data_label_np, h_output_squeeze_np)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(source_data_label_np, h_output_squeeze_np)
        average_precision = auc(recall, precision)


        top_k_acc_dict = {}
        for tmp_aim_k in [100, 500, 1000, 5000, 10000]:
            top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
            
        result[label_name] = {'loss':loss.item(), 'roc_auc':roc_auc, 'average_precision':average_precision,
                             'top_k_acc':top_k_acc_dict}

    return result

