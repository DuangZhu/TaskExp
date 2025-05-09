import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import cv2
from einops import rearrange
import time
import logging
import concurrent.futures
import copy

class TaskExp_Dataset(Dataset):
    def __init__(self, directory):
        """
        初始化数据集类。
        :param directory: 包含 .npz 文件的文件夹路径。
        """
        self.directory = directory
        self.data = []
        self.mean = torch.Tensor([0.0474, 0.171, 0.0007])
        self.std = torch.Tensor([0.4430, 0.3323, 0.7])
        # self.files = glob.glob(os.path.join(directory, '**', '*.pt'), recursive=True) # single directory
        self.files = self.threaded_search(directory) # many directory
        print('Data_size:', str(len(self)))
        
        
    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.files)
    
    def __getitem__(self, idx): 
        file_path = copy.deepcopy(self.files[idx])
        data = torch.load(file_path)
        input_data = data['input']
        label = data['label']  
        gt = data['GT']
        gt_value = data['Return']
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        input_data = (input_data - mean) / std
        mean = self.mean.view(3, 1, 1)
        std = self.std.view(3, 1, 1)
        gt = (gt - mean) / std
        return input_data, label, gt, gt_value
    
    def find_files(self, directory):
        return glob.glob(os.path.join(directory, '*.pt'))

    def threaded_search(self, base_directory):
        subdirectories = [d.path for d in os.scandir(base_directory)if d.is_dir() and d.name in {'1','2','3','4','5','6','7','8','9','10'}]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_directory = {executor.submit(self.find_files, directory): directory for directory in subdirectories}
            results = []
            for future in concurrent.futures.as_completed(future_to_directory):
                results.extend(future.result())
        return results


if __name__ == '__main__':
    dataset = TaskExp_Dataset(directory='/remote-home/ums_zhushaohao/new/2025/MAexp/test_make_data/ours')
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)



