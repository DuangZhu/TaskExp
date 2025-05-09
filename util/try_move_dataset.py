import shutil
import os
import glob

def max_pt_number(folder):
    """返回指定文件夹中最大的.pt文件编号"""
    max_number = 0
    for filename in os.listdir(folder):
        if filename.endswith('.pt'):
            number = int(filename.split('.')[0])
            max_number = max(max_number, number)
    return max_number

# 定义文件夹路径
source_folder = '/remote-home/share/zsh/6'
destination_folder = '/remote-home/share/zsh/mae_training_set_all'

# 确定文件夹a中的最大文件编号
max_number = max_pt_number(destination_folder)
# max_number = 35874

files = glob.glob(os.path.join(source_folder, '*.pt'))
image_files = sorted(files, key=lambda s: int(s.split('/')[-1].split('.')[0]))
for source_file in image_files:
    max_number += 1  # 为新文件增加编号
    new_filename = f"{max_number}.pt"  # 创建新的文件名
    destination_file = os.path.join(destination_folder, new_filename)
    # 移动并重命名文件
    shutil.move(source_file, destination_file)
