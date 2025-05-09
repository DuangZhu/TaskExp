import glob
import os
import torch
from einops import rearrange
import cv2
import numpy as np


def global_to_local(global_point, car_pos):
    '''
    Calculate the local position of a point in the global map
    '''
    map_real_w = 125
    map_real_h = 125
    # 首先，将全局点相对于汽车位置进行平移
    local_x = global_point[0] - car_pos[0]
    local_y = global_point[1] - car_pos[1]

    # 将转换后的局部坐标调整到智能体坐标系的中心
    center = torch.tensor([map_real_w / 2, map_real_h / 2])
    local_x += center[0]
    local_y += center[1]

    return torch.stack((local_x, local_y))

def local_to_global(local_point, car_pos):
    '''
    Calcluate the global position of the goal from local map
    '''
    map_w = 125
    map_h = 125
    center = torch.tensor([map_w / 2, map_h / 2])
    x = local_point[0] - center[0]
    y = local_point[1] - center[1]
    return torch.stack((car_pos[0] + x, car_pos[1] + y), dim=0)
# ------------multi file --------
# directory = '/remote-home/share/zsh/test_thecode'
# files = glob.glob(os.path.join(directory, '*.pt'))
# image_files = sorted(files, key=lambda s: int(s.split('/')[-1].split('.')[0]))
# for i, file_path in enumerate(image_files):
#     if i % 3 == 0:
#         data = torch.load(file_path)
#         input_data = data['input']
#         label = data['label']  # 确保键名与保存时一致
#         gt = data['GT']
#         point = global_to_local(label[-2:], label[:4])/125*128
#         image = torch.cat((input_data, gt.unsqueeze(1)), dim = 1)
#         point = torch.clamp(point, max=125-1e-5)
#         input_data[0, :, point[0].long(), point[1].long()] = 1
#         a = (rearrange(input_data[0], 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
#         cv2.imwrite('/remote-home/share/zsh/images/' + f'{i}.png', a)
#         gt[:, (label[-2]/125*128).long(), (label[-1]/125*128).long()] = 1
#         a = (rearrange(gt, 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
#         cv2.imwrite('/remote-home/share/zsh/images/' + f'{i}_.png', a)
    
    
# ------Single file--------------

def max_pt_number(folder):
    """返回指定文件夹中最大的.pt文件编号"""
    max_number = 0
    for filename in os.listdir(folder):
        if filename.endswith('.pt'):
            number = int(filename.split('.')[0])
            max_number = max(max_number, number)
    return max_number
dir = '/remote-home/share/zsh/smmr_dataset/random_0_smmr/16'
max_number = max_pt_number(dir)
file_path = dir + '/' + str(max_number) + '.pt'
# file_path = '/remote-home/share/zsh/smmr_dataset/TrainingSet/1/1.pt'
i = 0
print(file_path)
data = torch.load(file_path)
input_data = data['input']
label = data['label']  # 确保键名与保存时一致
gt = data['GT']
point = global_to_local(label[-2:], label[:4])/125*128
# point = global_to_local(label[-2:], label[:4])
# point = local_to_global(point, label[:4])
image = torch.cat((input_data, gt.unsqueeze(1)), dim = 1)
point = torch.clamp(point, max=125-1e-5)
input_data[0, :, point[0].long(), point[1].long()] = 1
a = (rearrange(input_data[0], 'c w h -> w h c').cpu().numpy()*255).astype(np.uint8)
cv2.imwrite('/remote-home/ums_zhushaohao/2023/mae/' + f'{i}.png', a)
