import dgl
import copy
import numpy as np 
import torch
from scipy import io
import os
import pickle
from collections import defaultdict


def load_dgl_data(config):
    name = config['dataset']
    if name == 'DBLP':
        return DBLP(config)
    elif name == 'Aminer':
        return Aminer(config)
    elif name == 'Logistic':
        from pub_data.Logistic_data_loader import DataLoader
        return DataLoader(config)
    else:
        raise NameError(f"Do not support the dataset {name}. Please try DBLP/Aminer/JD.")


class DBLP:
    def __init__(self, config):
        from pub_data.DBLP_data_loader import DataLoader
        data_path = 'Data/DBLP'
        data_loader = DataLoader(data_path)
        # load features
        node_type_ind = ['author', 'paper', 'phase', 'venue']
        feature = {}
        num_nodes = {}
        node_shift = {}
        self.node_type_to_feature_len_dict = {}
        for ind, name in enumerate(node_type_ind):
            feature[name] =  data_loader.nodes['attr'][ind]
            if feature[name] is None:
                feature[name] = torch.eye(data_loader.nodes['count'][ind])
            num_nodes[name] =  data_loader.nodes['count'][ind]
            node_shift[name] =  data_loader.nodes['shift'][ind]
            self.node_type_to_feature_len_dict[name] = feature[name].shape[1]
           
        # load adjs
        adj_dict = {}
        node_edge_tuple = [('author','a-p','paper'), ('paper','p-ph','phase'),
                          ('paper','p-v','venue'), ('paper','p-a','author'),
                          ('phase','ph-p','paper'), ('venue','v-p','paper')]
        for ind, name in enumerate(node_edge_tuple):
            head_node_shift = node_shift[name[0]]
            tail_node_shift = node_shift[name[-1]]
            adj_dict[name] = (data_loader.links['data'][ind][:,0]-head_node_shift, data_loader.links['data'][ind][:,1]-tail_node_shift)

        adj_dict[('author','a-a','author')] = (torch.arange(0, num_nodes['author']), torch.arange(0, num_nodes['author']))
        adj_dict[('paper','p-p','paper')] =  (torch.arange(0, num_nodes['paper']), torch.arange(0, num_nodes['paper']))
          
       # load label
        a_train_data = data_loader.labels_train['data'][node_shift['author']:node_shift['author']+num_nodes['author']]
        a_test_data = data_loader.labels_test['data'][node_shift['author']:node_shift['author']+num_nodes['author']]
        author_label = torch.FloatTensor(a_train_data + a_test_data)
        
        # add paper-venue label based on p-v link, then delete p-v link
        split_data_path = os.path.join(data_path, 'label_mask_split.npy')
        paper_label = torch.nn.functional.one_hot(adj_dict[('paper','p-v','venue')][1], num_classes=num_nodes['venue']).float()
        paper_label_num = num_nodes['paper']
        del adj_dict[('paper','p-v','venue')]
        del adj_dict[('venue','v-p','paper')]
        del num_nodes['venue']
        del feature['venue']
        
        # load label mask
        if os.path.exists(split_data_path):
            split_data = np.load(split_data_path, allow_pickle=True)
            split_data = split_data.tolist()
            author_mask, paper_mask = split_data[0], split_data[1]
        else:
            # paper mask
            train_ratio = 0.6
            val_ratio = 0.2
            idx = np.arange(0, paper_label_num)
            np.random.shuffle(idx)
            train_idx, val_idx, test_idx = np.array_split(idx, [int(train_ratio*paper_label_num), int((val_ratio+train_ratio)*paper_label_num)])
            mask = torch.zeros(paper_label_num, dtype=torch.bool)
            p_train_mask, p_val_mask, p_test_mask = copy.deepcopy(mask), copy.deepcopy(mask), copy.deepcopy(mask)
            p_train_mask[train_idx] = True
            p_val_mask[val_idx] = True
            p_test_mask[test_idx] = True
                
             
            # author mask
            a_train_mask = data_loader.labels_train['mask'][node_shift['author']:node_shift['author']+num_nodes['author']]
            a_test_mask = data_loader.labels_test['mask'][node_shift['author']:node_shift['author']+num_nodes['author']]
            val_ratio = 0.2
            train_idx = np.nonzero(a_train_mask)[0] #
            np.random.shuffle(train_idx)
            split = int(train_idx.shape[0]*val_ratio)
            a_train_mask_select = copy.deepcopy(a_train_mask)
            a_train_mask_select[train_idx[:split]] = False
            a_val_mask = copy.deepcopy(a_train_mask)
            a_val_mask[train_idx[split:]] = False

            author_mask = [a_train_mask_select, a_val_mask, a_test_mask]
            paper_mask = [p_train_mask, p_val_mask, p_test_mask]
            np.save(split_data_path, [author_mask, paper_mask], allow_pickle=True)

        author_mask  = [torch.BoolTensor(x) for x in author_mask]
        paper_mask  = [torch.BoolTensor(x) for x in paper_mask]

        # build graph
        hg_dgl = dgl.heterograph(adj_dict, num_nodes_dict = num_nodes)
        hg_dgl.ndata['feat'] = feature

        for ind, i in enumerate(['train', 'val', 'test']):
            hg_dgl.ndata[i+"_label"] = {'author': author_label, 'paper':paper_label}
            hg_dgl.ndata[i+"_label_mask"] = {'author': author_mask[ind], 'paper': paper_mask[ind]}

        self.hg_dgl = hg_dgl
        if config['args_cuda']:
            self.hg_dgl = self.hg_dgl.to(config['device'])
        
        self.node_dict = {}
        self.edge_dict = {}
        for ind, ntype in enumerate(hg_dgl.ntypes):
            self.node_dict[ntype] = ind
        for ind, etype in enumerate(hg_dgl.etypes):
            self.edge_dict[etype] = ind
            
        self.edge_type_each_node = defaultdict(list)
        for srctype, etype, dsttype in hg_dgl.canonical_etypes:
            self.edge_type_each_node[srctype].append(self.edge_dict[etype])



class Aminer:
    def __init__(self, config):
        data_path = 'Data/'
        split_data_path = os.path.join(data_path, 'Aminer_split.npy')
        full_data_path = os.path.join(data_path, 'Aminer.mat')
        data = io.loadmat(full_data_path)

        # load feature
        feature = {}
        num_nodes = {}
        self.node_type_to_feature_len_dict = {}
        feature['paper'] = torch.FloatTensor(normalization(data["PvsF"]))
        feature['author'] = torch.FloatTensor(normalization(data["AvsF"]))
        for node in feature:
            self.node_type_to_feature_len_dict[node] =  feature[node].shape[1]
        num_nodes['paper'] = data["PvsF"].shape[0]
        num_nodes['author'] = data["AvsF"].shape[0]

        # load adj
        adj_dict = {}
        adj_dict[('paper','p-a','author')] = convert_csc_to_value(data["PvsA"])
        adj_dict[('author','a-p','paper')] = convert_csc_to_value(data["PvsA"].transpose())
        adj_dict[('paper','p-p','paper')] = convert_csc_to_value(data["PvsP"])
        adj_dict[('author','a-a','author')] = convert_csc_to_value(data["AvsA"])
        
            
        author_label = torch.FloatTensor(data["AvsC"].toarray())
        paper_label = torch.FloatTensor(data["PvsC"].toarray())

        def sample_train_test_val(num, val_ratio=0.1, test_ratio=0.1):
            train_ratio = 1 - val_ratio - test_ratio
            idx = np.arange(0, num)
            np.random.shuffle(idx)
            train_idx, val_idx, test_idx = np.array_split(idx, [int(train_ratio*num), int((val_ratio+train_ratio)*num)])

            mask = torch.zeros(num, dtype=torch.bool)
            train_mask, val_mask, test_mask = copy.deepcopy(mask), copy.deepcopy(mask), copy.deepcopy(mask)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
            return train_mask, val_mask, test_mask

        
        if os.path.exists(split_data_path):
            split_data = np.load(split_data_path, allow_pickle=True)
            split_data = split_data.tolist()
            author_mask, paper_mask = split_data[0], split_data[1]
            author_mask  = [torch.tensor(x) for x in author_mask]
            paper_mask  = [torch.tensor(x) for x in paper_mask]
        else:
            # load label 
            val_ratio, test_ratio = 0.1, 0.1
            author_mask = sample_train_test_val(author_label.shape[0], val_ratio, test_ratio)
            paper_mask = sample_train_test_val(paper_label.shape[0], val_ratio, test_ratio)


            author_mask = [x.numpy() for x in author_mask]
            paper_mask = [x.numpy() for x in paper_mask]
            save_path = '../../Data/public_data/Aminer_split.npy'
            np.save(save_path, [author_mask, paper_mask], allow_pickle=True)


        # build graph
        hg_dgl = dgl.heterograph(adj_dict, num_nodes_dict = num_nodes)
        hg_dgl.ndata['feat'] = feature

        hg_dgl.ndata["train_label"] = {'author': author_label, 'paper': paper_label}
        hg_dgl.ndata["val_label"] = {'author': author_label, 'paper': paper_label}
        hg_dgl.ndata["test_label"] = {'author': author_label, 'paper': paper_label}
        hg_dgl.ndata["train_label_mask"] = {'author': author_mask[0], 'paper': paper_mask[0]}
        hg_dgl.ndata["val_label_mask"] = {'author': author_mask[1], 'paper': paper_mask[1]}
        hg_dgl.ndata["test_label_mask"] = {'author': author_mask[2], 'paper': paper_mask[2]}
        
         
        self.hg_dgl = hg_dgl
        if config['args_cuda']:
            self.hg_dgl = self.hg_dgl.to(config['device'])
        
        
        self.node_dict = {}
        self.edge_dict = {}
        for ind, ntype in enumerate(hg_dgl.ntypes):
            self.node_dict[ntype] = ind
        for ind, etype in enumerate(hg_dgl.etypes):
            self.edge_dict[etype] = ind
            
        self.edge_type_each_node = defaultdict(list)
        for srctype, etype, dsttype in hg_dgl.canonical_etypes:
            self.edge_type_each_node[srctype].append(self.edge_dict[etype])
        


    
def convert_csc_to_value(csc_matrix):
    coo_matrix = csc_matrix.tocoo()
    row, col = coo_matrix.row, coo_matrix.col
    return (row, col)

def normalization(data):
    mean = np.mean(data, 0, keepdims=True)
    std = np.std(data, 0, keepdims=True)
    norm_data = (data - mean) / std
    return norm_data



