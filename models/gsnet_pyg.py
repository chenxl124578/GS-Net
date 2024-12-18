import os
from math import ceil

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module,BatchNorm1d, Linear, Conv1d
import numpy as np

from torch_geometric.nn import DenseSAGEConv

from .src.chamfer_distance.chamfer_distance import ChamferDistance

class GNN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = BatchNorm1d(out_channels)

        if lin is True:
            self.lin = Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)   
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
    
        x1 = self.bn(1, self.conv1(x0, adj, mask)).relu()  
        x2 = self.bn(2, self.conv2(x1, adj, mask)).relu()
        x3 = self.bn(3, self.conv3(x2, adj, mask)).relu()
        
        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x

class GSNet(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, hidden_dim=128, embedding_dim=128,
        assign_ratio=1/32,linkpred=False):
        super().__init__()
        self.max_num_nodes=max_num_nodes
        self.linkpred = linkpred

        self.num_out_points = ceil(assign_ratio * max_num_nodes)
        self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)

        self.fc1 = nn.Linear(2 * hidden_dim + embedding_dim, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * self.num_out_points)   

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        

    def forward(self, x, adj, mask=None):
        x = self.gnn1_embed(x, adj, mask)
        x = torch.max(x, 1)[0]

        x = F.relu(self.bn_fc1(self.fc1(x))) 
        x = F.relu(self.bn_fc2(self.fc2(x))) 
        x = F.relu(self.bn_fc3(self.fc3(x))) 
        x = self.fc4(x) 

        x = x.view(-1,self.num_out_points, 3) 
        x = x.contiguous()

        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss
    
    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            
            return self.link_loss

        return torch.tensor(0).to(adj)
