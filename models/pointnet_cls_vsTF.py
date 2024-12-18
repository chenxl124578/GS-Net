"""
copy in Pointnet_Pointnet2_pytorch
create in 2022/3/29
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder_vsTF, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True, change_init=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder_vsTF(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, k)
        
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        print("now using PoingNet_vsTF")
        if change_init:   # 改初始化
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    m.weight.data = nn.init.xavier_uniform_(m.weight.data)  # weight xavier（TF源码是xavier_initializer(),应该就是uniform了）
                    if m.bias is not None:
                        m.bias.data = nn.init.constant(m.bias.data, 0.0) # bias全0
                if isinstance(m,nn.Linear):
                    m.weight.data = nn.init.xavier_uniform_(m.weight.data)  # weight 为 xavier 均匀分布
                    if m.bias is not None:
                        m.bias.data = nn.init.constant(m.bias.data, 0.0) # bias全0
            

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.dp1(F.relu(self.bn1(self.fc1(x))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        retrieval_vector = x   # for shape retrieval task, TF版里是先fc bn relu再dropout 这里先dropout可能会有不同
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat, retrieval_vector

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
