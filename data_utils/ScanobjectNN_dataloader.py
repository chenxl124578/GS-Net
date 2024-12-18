import os,sys

import numpy as np
import warnings
import pickle
import h5py
import torch

from tqdm import tqdm
from torch.utils.data import Dataset


class ScanobjectNNDataSet(Dataset):
    def __init__(self, root, num_point, split='train', dataclass=None):
        super().__init__()
        self.root = root
        self.npoint = num_point
        assert(split in ['train','test'])
        if split == "train":
            self.split = "training"
        else:
            self.split = "test"
        
        if dataclass == 'OBJ_BG':
            suffix_name = "_objectdataset.h5"
        elif dataclass == 'PB_T25':
            suffix_name = "_objectdataset_augmented25_norot.h5"
        elif dataclass == 'PB_T25_R':
            suffix_name = "_objectdataset_augmented25rot.h5"
        elif dataclass == 'PB_T50_R':
            suffix_name = "_objectdataset_augmentedrot.h5"
        elif dataclass == 'PB_T50_RS':
            suffix_name = "_objectdataset_augmentedrot_scale75.h5"
        else:
            raise ValueError
        """read txt"""
        self.DataFiles = os.path.join(root, self.split+suffix_name)
        self.data, self.label = self.load_h5py(self.DataFiles)

        # self.data = np.concatenate(data, axis=0)
        # self.label = np.concatenate(label, axis=0) 

        # if self.debug:   # if debug just use 128 sample to train or test
            # self.data = self.data[:128]
            # self.label = self.label[:128]
        
        self.data = self.data[:,:self.npoint]  # get num point,换成在这里采点能减少内存占用


    def load_h5py(self, h5_name):
        
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')  # 2309,2048,3  581,2048,3
        label = f['label'][:].astype('int64')  # 2309,  581,
        f.close()
            
        return data, label



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
    modelnet = ScanobjectNNDataSet(root="../data/scanobjectnn/h5_files/main_split/",split="train",num_point=1024,dataclass='OBJ_BG')
    dataloader = torch.utils.data.DataLoader(modelnet,batch_size=8)
    print(modelnet.__len__())  # train:2309  test:581
    # for batch_id, (points,target) in enumerate(dataloader):
    #     print(points.shape) # 8,1024,3
    #     print(target.shape) # 8,1
