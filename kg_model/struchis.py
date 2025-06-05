# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import copy
import dgl
from collections import defaultdict
import math
from torch_scatter import scatter, scatter_softmax

from IPython.core.debugger import set_trace
from .core import DNN, PredictionLayer



class StrucHIS(nn.Module):

    def __init__(self, config):
        super(StrucHIS, self).__init__()
        
        self.num_levels = config['num_levels']
        self.task_names = config['task_name']
        self.num_tasks = len(self.task_names)
        self.specific_expert_num = 1
        self.config = config
        device = config['device']
        
        self.edge_type_each_node = config['edge_type_each_node']
        self.edge_num_node = {} 
        for node in self.edge_type_each_node:
            self.edge_num_node[node] = len(self.edge_type_each_node[node])

        input_dim = config['node_feature_hid_len']
        no_level0_input_dim = config['GAT_hid_len']
        self.input_dim = [input_dim * 2 for i in range(self.num_levels)]  
        self.no_level0_input_dim = [no_level0_input_dim*2 for i in range(self.num_levels)] 
        

        # 0. feature embedding net
        self.edge_list = list(config['edge_dict'].keys())
        node_dict =  config['node_dict']
        node_type_to_feature_len_dict = config['node_type_to_feature_len_dict']
        node_feature_hid_len = config['node_feature_hid_len']
        self.Node_Transform_list = {}
        for tmp_node_type in node_dict:
            tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
            self.Node_Transform_list[tmp_node_type] = tmp_linear
            self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])

        
        def multi_module_list(num_level, num_tasks, expert_num, inputs_dim_level0, inputs_dim_not_level0, hidden_units):
            return nn.ModuleList(
                [nn.ModuleList([nn.ModuleList([DNN(inputs_dim_level0[level_num] if level_num == 0 else inputs_dim_not_level0[level_num],
                                                   hidden_units, device=device) for _ in
                                               range(expert_num)])
                                for _ in range(num_tasks)]) for level_num in range(num_level)])
        def multi_module_list_hetegraph(num_level, num_tasks, expert_num, config, type):
            return nn.ModuleList(
                [nn.ModuleList([HGB_MTL_Layer(config, 'share') if type=='share' else HGB_MTL_Layer(config, self.task_names[i]) 
                                for _ in range(expert_num)])
                                for i in range(num_tasks)])

        # 1. experts
        # task-specific experts
        self.specific_experts = multi_module_list_hetegraph(self.num_levels, self.num_tasks, self.specific_expert_num, config, 'specific')

        # 2. gates
        self.specific_gate_dnn = nn.ModuleDict()
        self.specific_gate_dnn_final_layer = nn.ModuleDict()
        self.gate_dnn_hidden_units = [64]
        for node in node_dict:
            specific_gate_output_dim = [(self.num_tasks * self.specific_expert_num) * len(self.edge_type_each_node[node]) for i in range(self.num_levels)] 
            if len(self.gate_dnn_hidden_units) > 0:
                self.specific_gate_dnn[node] = multi_module_list(self.num_levels, self.num_tasks, 1,
                                                           self.input_dim, self.no_level0_input_dim,
                                                           self.gate_dnn_hidden_units)
            self.specific_gate_dnn_final_layer[node] = nn.ModuleList(
            [nn.ModuleList([nn.Linear(
                self.gate_dnn_hidden_units[-1] if len(self.gate_dnn_hidden_units) > 0 else self.input_dim[level_num] if level_num == 0 else
                self.no_level0_input_dim[level_num], specific_gate_output_dim[level_num], bias=False)
                for _ in range(self.num_tasks)]) for level_num in range(self.num_levels)])


        # 3. tower dnn (task-specific)
        self.tower_dnn_final_layer = nn.ModuleList([HGB_final(config, i)  for i in range(self.num_tasks)])
                
        self.to(device)

    # a single cgc Layer
    def cgc_net(self, inputs, level_num, G):
        # inputs: [task1, task2, ... taskn]
        
        if level_num > 0:
            inputs, all_type_edge_src_node_feature_adj_dict = inputs  
            
        # 1. experts
        # task-specific experts
        input_target_node_feature_all = []
        specific_expert_outputs = []
        expert_graph = []
        for i in range(self.num_tasks):
            specific_expert_outputs_i, expert_graph_i, input_target_node_feature_all_i = [], [], []
            for j in range(self.specific_expert_num):
                if level_num == 0:
                    specific_expert_output, specific_input_target_node_feature, specific_graph = self.specific_experts[i][j](inputs[i], level_num)
                else:
                    specific_expert_output, specific_input_target_node_feature, specific_graph = self.specific_experts[i][j](G, level_num, inputs[i], all_type_edge_src_node_feature_adj_dict[i][j])
                specific_expert_outputs_i.append(specific_expert_output)
                expert_graph_i.append(specific_graph)
                input_target_node_feature_all_i.append(specific_input_target_node_feature)

            # feature after HGB layer
            specific_expert_outputs.append(specific_expert_outputs_i)
            # graph structure info
            expert_graph.append(expert_graph_i)
            # feature before HGB layer
            input_target_node_feature_all.append(input_target_node_feature_all_i)

            
        # 2. gates
        # gates for task-specific experts
        cgc_outs = []
        for i in range(self.num_tasks):
            cgc_outs_i = {}
            all_gate_output = {}
            for node_type in input_target_node_feature_all[i][0]:
                input_target_node_feature = input_target_node_feature_all[i][0][node_type]
                # concat task-specific expert
                cur_experts_outputs = [specific_expert_outputs_i_j[node_type][0] for specific_expert_outputs_i in specific_expert_outputs 
                                       for specific_expert_outputs_i_j in specific_expert_outputs_i]
                a_relation = specific_expert_outputs[i][0][node_type][1]
                gate_input_feature, h_head, edge_list, h_e, tmp_edge, res, shp = specific_expert_outputs[i][0][node_type][2]
                head_ind, _ = edge_list
                leakyrelu_func, dropout_func, act_func = specific_expert_outputs[i][0][node_type][3]

                # [N, n_edge_type, d] -> [N, n_edge_type, tasks, d]
                cur_experts_outputs = torch.stack(cur_experts_outputs, -2) # stack in column
                # gate dnn
                # input_shared_target_node_feature: [N, d]
                if len(self.gate_dnn_hidden_units) > 0:
                    gate_dnn_out = self.specific_gate_dnn[node_type][level_num][i][0](gate_input_feature)
                    gate_dnn_out = self.specific_gate_dnn_final_layer[node_type][level_num][i](gate_dnn_out)
                else:
                    gate_dnn_out = self.specific_gate_dnn_final_layer[node_type][level_num][i](gate_input_feature)

                # gate_dnn_out: # [N_noagg, n_edge_type*tasks] -> [N_noagg, n_edge_type, tasks]
                gate_dnn_out = gate_dnn_out.reshape(gate_dnn_out.shape[0], len(self.edge_type_each_node[node_type]), -1)
                gate_dnn_out = gate_dnn_out.softmax(-1)
                all_gate_output[node_type] = gate_dnn_out.clone()
                
                # gate_dnn_out: [N_noagg, n_edge_type, tasks]
                # cur_experts_outputs: [N_noagg, n_edge_type, tasks, d]
                # [N_noagg, n_edge_type, tasks, 1] * [N_noagg, n_edge_type, tasks, d] -> [N_noagg, n_edge_type, tasks, d]
                gate_mul_expert = gate_dnn_out.unsqueeze(-1) * cur_experts_outputs
                # Sum over tasks dimension -> [N_noagg, n_edge_type, d]
                gate_mul_expert = gate_mul_expert.sum(dim=2)

                # attention aggregate differnt types of relation features
                # [N, 1, d], [N, n_edge_type, d] -> [N, n_edge_type, 2d]
                concat_features = torch.cat([h_head.expand(-1, gate_mul_expert.size(1), -1), 
                                          gate_mul_expert], dim=-1)
 
                # [N, n_edge_type, 2d] -> [N, n_edge_type, 1]
                attention_scores = a_relation(concat_features)
                attention_weights = torch.softmax(attention_scores, dim=1)
                # [N, n_edge_type, 1] * [N, n_edge_type, d] -> [N, n_edge_type, d] -> [N, d]
                out = (attention_weights * gate_mul_expert).sum(dim=1)

                # node residual
                gate_mul_expert = out + res 
                # use activation or not
                if act_func is not None:
                    gate_mul_expert = act_func(gate_mul_expert)
                        

                # cur_experts_outputs 
                cgc_outs_i[node_type] = gate_mul_expert
            cgc_outs.append(cgc_outs_i)


        

        return cgc_outs, expert_graph
    
    
    
    
  
    
    def forward(self, X):
        
        for ntype in X.ntypes:
            X.nodes[ntype].data['feat_emb'] = self.Node_Transform_list[ntype]((X.nodes[ntype].data['feat']))
      
        # repeat `X` for several times to generate cgc input
        model_inputs = [X] * (self.num_tasks + 1)  # [task1, task2, ... taskn]
        model_outputs = []
        for i in range(self.num_levels):
            model_outputs = self.cgc_net(inputs=model_inputs, level_num=i, G=X)
            model_inputs = model_outputs
            

        # tower dnn (task-specific)
        model_outputs, specific_expert_graph = model_outputs
        task_outs = {}
        select_expert = 0
        for i in range(self.num_tasks):
            output = self.tower_dnn_final_layer[i](model_outputs[i], specific_expert_graph[i][select_expert], X, i)
            task_outs[self.task_names[i]] = output
           
        return task_outs









class HGB_final(nn.Module):
    def __init__(self, config, task_ind):
        super().__init__()
        
        node_dict =  config['node_dict']
        edge_dict = config['edge_dict']
        Target_Node_Type = config['Target_Node_Type']
        node_type_to_feature_len_dict = config['node_type_to_feature_len_dict']
        
        self.Target_Node_Type = Target_Node_Type
        if type(self.Target_Node_Type) != list:
            self.Target_Node_Type = [self.Target_Node_Type]
        self.all_relation_list = list(edge_dict.keys())
        self.layer_num = config['layer_num']
        self.args_cuda = config['args_cuda']
        self.class_num = config['class_num'][task_ind]
        self.task_type = config['task_type']
        self.task_names = config['task_name']
        if len(self.task_names)>1 and len(self.Target_Node_Type)==1:
            self.single_node_mtl = True
        else:
            self.single_node_mtl = False
        node_feature_hid_len = config['node_feature_hid_len']
        GAT_hid_len = config['GAT_hid_len']
        
        
        self.edge_GAT = SimpleHGNLayer(edge_feats_len = config['edge_feats_len'], 
                                       in_features_len = GAT_hid_len ,
                                       out_features_len = self.class_num,
                                       node_dict = node_dict,
                                       edge_dict = self.all_relation_list,
                                       edge_type_each_node = config['edge_type_each_node'],
                                       node_residual=True,
                                       activation=None,
                                       spilt_edge=False
                                      )

        self.activation = nn.Softmax(1)
        
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.edge_GAT.reset_parameters()

    def forward(self, gate_weighted_feaure, all_type_edge_src_node_feature_adj_dict, G, target_node_index):  
        
        for srctype in all_type_edge_src_node_feature_adj_dict:
            all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'] = []
            
        # update node features by gate merging results
        for srctype, etype, dsttype in G.canonical_etypes:
            all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = gate_weighted_feaure[srctype]
            tail_feature = gate_weighted_feaure[dsttype]
            all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)

        inter_srctype_feaure = {}
        # apply hetegeneous graph conv layer for each type of source node
        for srctype in all_type_edge_src_node_feature_adj_dict:
            inter_srctype_feaure[srctype] = self.edge_GAT(all_type_edge_src_node_feature_adj_dict[srctype], all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'],srctype=srctype)

            
        # prediction
        if self.single_node_mtl:
            target_node = self.Target_Node_Type[0]
        else:
            target_node = self.Target_Node_Type[target_node_index]
        h_prime = inter_srctype_feaure[target_node][0]
        h_prime = inter_srctype_feaure[target_node][0]

        # L2 Normalization
        h_prime = h_prime / (torch.max(torch.norm(h_prime, dim=1, keepdim=True), self.epsilon))

        if self.single_node_mtl:
            task_name = self.task_names[target_node_index]
            task_type = self.task_type[task_name]
        else:
            task_type = self.task_type[target_node]
            
        if task_type == 'single-label':
            output = nn.Softmax(1)(h_prime)
        elif task_type == 'multi-label':
            output = torch.sigmoid(h_prime)
        else:
            raise NameError(f'task type {self.task_type[target_node]} does not exist!')

        if self.class_num<=2:
            output = output[:,0]

        return output
    

    
    
    

class HGB_MTL_Layer(nn.Module):
    def __init__(self, config, task_name):
        super().__init__()
        
        node_dict = config['node_dict']
        edge_dict = config['edge_dict']
        Target_Node_Type = config['Target_Node_Type']
        node_type_to_feature_len_dict = config['node_type_to_feature_len_dict']
        
        self.Target_Node_Type = Target_Node_Type
        if type(self.Target_Node_Type) != list:
            self.Target_Node_Type = [self.Target_Node_Type]
        self.all_relation_list = list(edge_dict.keys())
        self.layer_num = config['layer_num']
        self.args_cuda = config['args_cuda']
        self.task_name = task_name
        node_feature_hid_len = config['node_feature_hid_len']
        GAT_hid_len = config['GAT_hid_len']
        
      
        self.edge_GAT =  nn.ModuleList()
        # input projection
        spilt_edge = True
        self.edge_GAT.append(SimpleHGNLayer(edge_feats_len = config['edge_feats_len'], 
                                       in_features_len = node_feature_hid_len,
                                       out_features_len = GAT_hid_len,
                                       node_dict = node_dict,
                                       edge_dict = self.all_relation_list,
                                       edge_type_each_node = config['edge_type_each_node'],
                                       node_residual=False,
                                       activation=nn.functional.elu,
                                       spilt_edge = spilt_edge
                                      ))
        # middle projection
        for i in range(self.layer_num-1):
            self.edge_GAT.append(SimpleHGNLayer(edge_feats_len = config['edge_feats_len'], 
                                           in_features_len = GAT_hid_len,
                                           out_features_len = GAT_hid_len,
                                           node_dict = node_dict,
                                           edge_dict = self.all_relation_list,
                                           edge_type_each_node = config['edge_type_each_node'],
                                           node_residual=True,
                                           activation=nn.functional.elu,
                                           spilt_edge = spilt_edge           
                                          ))
        

        self.activation = nn.Softmax(1)
        
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        
        self.reset_parameters()

    def reset_parameters(self):
        for edge_GAT_i in self.edge_GAT:
            edge_GAT_i.reset_parameters()

    def forward(self, G, level_num, inter_srctype_feaure=None, all_type_edge_src_node_feature_adj_dict=None):
        
        if level_num == 0:
            h = {}

            for ntype in G.ntypes:
                h[ntype] = G.nodes[ntype].data['feat_emb'].clone()
            
            # create a data structure for HGB learning
            all_type_edge_src_node_feature_adj_dict = {}
            adj_begin_index = {}
            for srctype, etype, dsttype in G.canonical_etypes:
                if srctype not in all_type_edge_src_node_feature_adj_dict:
                    all_type_edge_src_node_feature_adj_dict[srctype] = defaultdict(list)
                    all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = h[srctype]
                    all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'] = None
                    adj_begin_index[srctype] = 0

                tail_feature = h[dsttype]
                adj = G.all_edges(etype=etype) 
                adj = torch.stack(list(adj), dim=0)
                all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)
                # modify node index 
                adj[1,:] = adj[1,:] + adj_begin_index[srctype]
                adj_begin_index[srctype] = adj_begin_index[srctype] + tail_feature.shape[0]
                all_type_edge_src_node_feature_adj_dict[srctype]['Adj'].append(adj)
                # index for edge type
                tmp_edge_index = self.all_relation_list.index(etype)
                tmp_edge_index = torch.tensor([tmp_edge_index]*adj.shape[1])
                if self.args_cuda:
                    tmp_edge_index = tmp_edge_index.cuda()
                all_type_edge_src_node_feature_adj_dict[srctype]['tmp_edge_index'].append(tmp_edge_index)

            feature_before_agg = h
        else:

            feature_before_agg = inter_srctype_feaure
            # update node features by gate merging results
            for srctype, etype, dsttype in G.canonical_etypes:
                all_type_edge_src_node_feature_adj_dict[srctype]['head_feature'] = inter_srctype_feaure[srctype]
                tail_feature = inter_srctype_feaure[dsttype]
                all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'].append(tail_feature)

        
        inter_srctype_feaure = {}
        # apply single hetegeneous graph conv layer for each type of source node
        for srctype in all_type_edge_src_node_feature_adj_dict:
            inter_srctype_feaure[srctype] = self.edge_GAT[level_num](all_type_edge_src_node_feature_adj_dict[srctype], all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'], srctype, level_num, self.task_name)
            all_type_edge_src_node_feature_adj_dict[srctype]['tail_feature'] = []
        # update edge weight residual
        for srctype, etype, dsttype in G.canonical_etypes:
            all_type_edge_src_node_feature_adj_dict[srctype]['res_attn'] = inter_srctype_feaure[srctype][1]

            
        for srctype in inter_srctype_feaure:
            inter_srctype_feaure[srctype] = inter_srctype_feaure[srctype][0]
     

        return inter_srctype_feaure, feature_before_agg, all_type_edge_src_node_feature_adj_dict
    
    
 
class SimpleHGNLayer(nn.Module):

    def __init__(
        self,
        edge_feats_len,
        in_features_len,
        out_features_len,
        node_dict,
        edge_dict,
        edge_type_each_node,
        feat_drop=0.5, 
        attn_drop=0.5,
        negative_slope=0.2, 
        node_residual=False,
        edge_residual_alpha=0.,
        activation=None,
        spilt_edge=False
        
    ):
        super(SimpleHGNLayer, self).__init__()
        self.edge_feats_len = edge_feats_len
        self.in_features_len = in_features_len
        self.out_features_len = out_features_len
        self.edge_dict = edge_dict
        self.edge_emb = nn.Parameter(torch.zeros(size=(len(edge_dict), edge_feats_len)))  # nn.Embedding(num_etypes, edge_feats)
        self.edge_type_each_node = edge_type_each_node

        self.W = nn.ModuleDict({
            ntype: nn.Linear(in_features_len, out_features_len, bias=False)
            for ntype in node_dict
        })
        # self.W = nn.Parameter(torch.FloatTensor(in_features_len, out_features_len))
        self.W_e = nn.Parameter(torch.FloatTensor(edge_feats_len, edge_feats_len))

        self.a_l = nn.Parameter(torch.zeros(size=(1, 1, out_features_len)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, 1, out_features_len)))
        self.a_e = nn.Parameter(torch.zeros(size=(1, 1, edge_feats_len)))

        self.a_relation = nn.Linear(2 * out_features_len, 1)

        self.feat_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        
        self.act = activation
        self.spilt_edge = spilt_edge

        if node_residual:
            self.node_residual = nn.Linear(in_features_len, out_features_len)
        else:
            self.register_buffer("node_residual", None)
        self.reset_parameters()
        self.edge_residual_alpha = edge_residual_alpha  # edge residual weight

    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l)
        reset(self.a_r)
        reset(self.a_e)

        for ntype in self.W:
            reset(self.W[ntype].weight)
        reset(self.W_e)
        reset(self.edge_emb)
    
    def forward(self, all_type_edge_src_node_feature_adj_dict, res_attn, srctype=None, level_num=None, task_name=None):
        
        head_node_feature = all_type_edge_src_node_feature_adj_dict['head_feature']
        all_tail_node_feature = all_type_edge_src_node_feature_adj_dict['tail_feature']  # [n1+n2+.., d]
        all_tail_node_feature = torch.cat(all_tail_node_feature, 0)
        edge_list = all_type_edge_src_node_feature_adj_dict['Adj']  # [2, n1+n2+..]
        edge_list = torch.cat(edge_list, 1)
        tmp_edge = all_type_edge_src_node_feature_adj_dict['tmp_edge_index']
        tmp_edge = torch.cat(tmp_edge)
        all_tail_node_num = all_tail_node_feature.shape[0]
        
        # d:in_features_len  D:out_features_len, de:out_features_len
        x_head = self.feat_drop(head_node_feature)  # [N, d]  
        x_tail = self.feat_drop(all_tail_node_feature)
        #  [N, d]*[d, D] -> [N, 1, D]
        # h_head = torch.matmul(x_head, self.W[srctype]).view(-1, 1, self.out_features_len) 
        # h_tail = torch.matmul(x_tail, self.W[srctype]).view(-1, 1, self.out_features_len) 
        h_head = torch.matmul(x_head, self.W[srctype].weight.T).view(-1, 1, self.out_features_len) 
        h_tail = torch.matmul(x_tail, self.W[srctype].weight.T).view(-1, 1, self.out_features_len) 
        # [edge_num, de]*[de, de] -> [edge_num, 1, de]
        e = torch.matmul(self.edge_emb, self.W_e).view(-1, 1, self.edge_feats_len)
        
        head_ind, tail_ind = edge_list
        h_e = (self.a_e * e).sum(dim=-1)[tmp_edge]
        
        if self.spilt_edge:

            # Self-attention on the nodes - Shared attention mechanism
            # [1, 1, D]*[N, 1, D] -> [N, 1] -> [sub_n, 1]
            h_l = (self.a_l * h_head).sum(dim=-1)[head_ind]
            h_r = (self.a_r * h_tail).sum(dim=-1)[tail_ind]
            edge_attention = self.leakyrelu(h_l + h_r + h_e) 
            # Cannot use dropout on sparse tensor , put dropout operation before sparse
            edge_attention = self.dropout(edge_attention)

            unique_edge_types = torch.unique(tmp_edge)
            out_by_type = []
            edge_attention_weight = []
            
            for edge_type in unique_edge_types:
                edge_mask = (tmp_edge == edge_type)
                cur_head_ind = head_ind[edge_mask]
                cur_tail_ind = tail_ind[edge_mask]
                cur_attention = edge_attention[edge_mask]
                
                cur_edge_list = torch.stack([cur_head_ind, cur_tail_ind])
                edge_attention_n = torch.sparse.FloatTensor(cur_edge_list, cur_attention[..., 0], (x_head.shape[0], x_tail.shape[0]))
                edge_attention_n = torch.sparse.softmax(edge_attention_n, dim=1)
                edge_attention_weight.append(edge_attention_n)
                
                # [N_head, N_tail]*[N_tail, D] -> [N_head, D]
                cur_out = torch.sparse.mm(edge_attention_n, h_tail[:,0,:])
                out_by_type.append(cur_out)

            # [num_head_nodes, num_edge_types, hidden_dim]
            out = torch.stack(out_by_type, dim=-2)  

            
            if self.node_residual is not None:
                # [N, d]*[d, D] -> [N,D]
                res = self.node_residual(head_node_feature)
            else:
                res = torch.zeros_like(head_node_feature)
                

            gate_input_feature = torch.cat([x_head, x_head], -1)
            #gate_input_feature = torch.cat([head_node_feature[head_ind, :], all_tail_node_feature[tail_ind, :]], -1)
            return (out, self.a_relation, (gate_input_feature, h_head, edge_list, h_e, tmp_edge, res,  (x_head.shape[0], x_tail.shape[0])), (self.leakyrelu, self.dropout, self.act)), None
        
        
        # Self-attention on the nodes - Shared attention mechanism
        # [1, 1, D]*[N, 1, D] -> [N, 1] -> [sub_n, 1]
        h_l = (self.a_l * h_head).sum(dim=-1)[head_ind]
        h_r = (self.a_r * h_tail).sum(dim=-1)[tail_ind]
        edge_attention = self.leakyrelu(h_l + h_r + h_e) 
        # Cannot use dropout on sparse tensor , put dropout operation before sparse
        edge_attention = self.dropout(edge_attention)

        # get aggregatin result by sparse matrix
        edge_attention_weight = []
        # [sub_n] -> [N_head, N_tail]
        edge_attention_n = torch.sparse.FloatTensor(edge_list, edge_attention[..., 0], (x_head.shape[0], x_tail.shape[0]))
        edge_attention_n = torch.sparse.softmax(edge_attention_n, dim=1)
        edge_attention_weight.append(edge_attention_n)
        
        out = torch.sparse.mm(edge_attention_n, h_tail[:,0,:])
        
        # node residual
        if self.node_residual is not None:
            # [N, d]*[d, D] -> [N,D]
            res = self.node_residual(head_node_feature)
            out += res
        # use activation or not
        if self.act is not None:
            out = self.act(out)

        return out, edge_attention_weight
    
    
    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features_len) + " -> " + str(self.out_features_len) + ")"

    