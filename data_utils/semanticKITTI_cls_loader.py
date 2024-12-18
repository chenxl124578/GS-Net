"""
By Xiaolei Chen
semantickitti_cls dataloader
"""

import os,sys

import numpy as np
import warnings
import pickle
import h5py
import torch

from tqdm import tqdm
from torch.utils.data import Dataset


class SemanticKITTI_cls_DataSet(Dataset):
    def __init__(self, root, split='train',keep_reflect=True,not_debug=False):
        super().__init__()
        self.root = root
        self.split = split
        self.debug = not not_debug
        assert(split in ['train','test'])
        assert(not_debug in [True,False])

        """read txt"""
        self.DataFilesPath = os.path.join(root, self.split+"_files.txt")
        self.DataLablesPath = os.path.join(root, self.split+"_lables.txt")
        self.DataFiles = [line.rstrip() for line in open(self.DataFilesPath)]
        

        data = self.load_data(self.DataFiles,keep_reflect)
        lables = [line.rstrip() for line in open(self.DataLablesPath)]

        self.data = np.array(data)
        self.label = np.array(lables) 

        if self.debug:   # if debug just use 128 sample to train or test
            self.data = self.data[:128]
            self.label = self.label[:128]

    def load_data(self, files_path, keep_reflect=True):
        all_data = []
        for file in files_path:
            data = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
            if not keep_reflect:
                data = data[:,:3]

            # data = data[:1024,:] # debug，后续样本完善后注释掉

            all_data.append(data)
            
        return all_data



    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # data, label = self.data[index][:self.npoint],self.label[index][:self.npoint]
        data, label = self.data[index],self.label[index] # debug，label不需要处理

        data = torch.from_numpy(data)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        return data, label


if __name__=='__main__':
    # modelnet = ModelNetDataSet(root="/disk2/cxl/code/SampleNet-pytorch/data/modelnet40_ply_hdf5_2048",num_point=1024)
    # dataloader = torch.utils.data.DataLoader(modelnet,batch_size=8)
    # print(modelnet.__len__())  # 9840个样本
    # for batch_id, (points,target) in enumerate(dataloader):
    #     print(points.shape) # 8,1024,3
    #     print(target.shape) # 8,1


    root = "/data/instance_data/SemanticKITTI_cls"
    train_files_txt = '/data/instance_data/train_files_fps1024.txt'
    test_files_txt = '/data/instance_data/test_files_fps1024.txt'
    train_lables_txt = '/data/instance_data/train_lables.txt'
    test_lables_txt = '/data/instance_data/test_lables.txt'

    semantickitti = SemanticKITTI_cls_DataSet(root=root)
    dataloader = torch.utils.data.DataLoader(semantickitti,batch_size=8)
    print(semantickitti.__len__())  # 9840个样本
    for batch_id, (points,target) in enumerate(dataloader):
        print(points.shape) # 8,1024,3
        print(target.shape) # 8
        print(target)