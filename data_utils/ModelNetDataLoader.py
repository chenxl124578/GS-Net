import os,sys

import numpy as np
import warnings
import pickle
import h5py
import torch

from tqdm import tqdm
from torch.utils.data import Dataset


class ModelNetDataSet(Dataset):
    def __init__(self, root, num_point, split='train',not_debug=False):
        super().__init__()
        self.root = root
        self.npoint = num_point
        self.split = split
        self.debug = not not_debug
        assert(split in ['train','test'])
        assert(not_debug in [True,False])

        """read txt"""
        self.DataFilesPath = os.path.join(root, self.split+"_files.txt")
        self.DataFiles = [line.rstrip() for line in open(self.DataFilesPath)]
        data, label = self.load_h5py(self.DataFiles)

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 

        if self.debug:   # if debug just use 127 sample to train or test
            self.data = self.data[:127]
            self.label = self.label[:127]
        
        self.data = self.data[:,:self.npoint]  # get num point,换成在这里采点能减少内存占用


    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            
        return all_data, all_label



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
    modelnet = ModelNetDataSet(root="/home1/qiusm/cxl/code/SampleNet-pytorch/data/modelnet40_ply_hdf5_2048",num_point=1024)
    dataloader = torch.utils.data.DataLoader(modelnet,batch_size=8)
    print(modelnet.__len__())  # 9840个样本
    for batch_id, (points,target) in enumerate(dataloader):
        print(points.shape) # 8,1024,3
        print(target.shape) # 8,1
