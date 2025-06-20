import os
import random
from torch.utils.data import Sampler, Dataset # 引用Dataset用于类型提示
from glob import glob
import numpy as np
import torch
class SinglePointCloudDataset(Dataset):
    def __init__(self, data_info_dir, some_npy_root_dir):
        self.all_npy_files_with_info = [] # [(filepath, point_count, class_label), ...]
        bucket_info_files = glob(os.path.join(data_info_dir, "*-*.txt"))
        for bucket_file_path in bucket_info_files:
            # ... 解析 point_count ...
            with open(bucket_file_path, 'r') as f:
                npy_filenames = [line.strip() for line in f if line.strip()]
                for npy_filename in npy_filenames:
                    point_count,class_label,npy_filename = self._parse_class_from_filename(npy_filename) # 你需要实现这个
                    self.all_npy_files_with_info.append(
                        {'path': os.path.join(some_npy_root_dir, npy_filename), # 假设文件名是相对的
                         'point_count': point_count,
                         'class': class_label,
                         'original_filename': npy_filename
                        }
                    )
        # 为每个样本创建一个唯一ID（即它在 all_npy_files_with_info 中的索引）
        self.file_to_id = {info['original_filename']: i for i, info in enumerate(self.all_npy_files_with_info)}
        self.id_to_file_info = {i: info for i, info in enumerate(self.all_npy_files_with_info)}


    def __len__(self):
        return len(self.all_npy_files_with_info)

    def __getitem__(self, idx):
        file_info = self.id_to_file_info[idx]
        npy_path = file_info['path']
        point_cloud_data = np.load(npy_path)
        # ... 其他处理 ...
        correspondences = torch.arange(file_info['point_count'], dtype=torch.long).unsqueeze(1).repeat(1, 2)
        return point_cloud_data, file_info['class'],file_info[ 'original_filename'] # 或者其他你需要返回的

    def _parse_class_from_filename(self, npy_filename):
        # 例如 xxx-zz-yyyyyy.npy -> zz
        try:
            return  int(npy_filename.split('-')[0]),int(npy_filename.split('-')[1]), int(npy_filename.split('-')[2])
        except:
            return -1 # 或者抛出错误
        

class SiamesePointCloudDataset(Dataset):
    def __init__(self, data_info_dir1,data_info_dir2, some_npy_root_dir):
        self.all_npy_files_with_info = [] # [(filepath, point_count, class_label), ...]
        bucket_info_files1 = glob(os.path.join(data_info_dir1, "*-*.txt"))
        bucket_info_files2 = glob(os.path.join(data_info_dir2, "*-*.txt"))
        for bucket_file_path in bucket_info_files1:
            # ... 解析 point_count ...
            with open(bucket_file_path, 'r') as f:
                npy_filenames = [line.strip() for line in f if line.strip()]
                for npy_filename in npy_filenames:
                    # 假设 npy_filename 本身就是可以直接加载的路径
                    # 或者你需要根据 npy_filename 和一个根目录构造完整路径
                    # 假设类别可以从 npy_filename 解析或通过其他方式获得
                    point_count,class_label,npy_filename = self._parse_class_from_filename(npy_filename) # 你需要实现这个
                    self.all_npy_files_with_info.append(
                        {'path': os.path.join(some_npy_root_dir, npy_filename), # 假设文件名是相对的
                         'point_count': point_count,
                         'class': class_label,
                         'original_filename': npy_filename
                        }
                    )
        for bucket_file_path in bucket_info_files2:
            # ... 解析 point_count ...
            with open(bucket_file_path, 'r') as f:
                npy_filenames = [line.strip() for line in f if line.strip()]
                for npy_filename in npy_filenames:
                    # 假设 npy_filename 本身就是可以直接加载的路径
                    # 或者你需要根据 npy_filename 和一个根目录构造完整路径
                    # 假设类别可以从 npy_filename 解析或通过其他方式获得
                    point_count,class_label,npy_filename = self._parse_class_from_filename(npy_filename) # 你需要实现这个
                    self.all_npy_files_with_info.append(
                        {'path': os.path.join(some_npy_root_dir, npy_filename), # 假设文件名是相对的
                         'point_count': point_count,
                         'class': class_label,
                         'original_filename': npy_filename
                        }
                    )
        # 为每个样本创建一个唯一ID（即它在 all_npy_files_with_info 中的索引）
        self.file_to_id = {info['original_filename']: i for i, info in enumerate(self.all_npy_files_with_info)}
        self.id_to_file_info = {i: info for i, info in enumerate(self.all_npy_files_with_info)}


    def __len__(self):
        return len(self.all_npy_files_with_info)

    def __getitem__(self, idx):
        file_info = self.id_to_file_info[idx]
        npy_path = file_info['path']
        point_cloud_data = np.load(npy_path)
        #correspondences = torch.arange(num_points, dtype=torch.long).unsqueeze(1).repeat(1, 2)
        # ... 其他处理 ...
        return point_cloud_data, file_info['class'],file_info[ 'original_filename'] # 或者其他你需要返回的

    def _parse_class_from_filename(self, npy_filename):
        # 例如 xxx-zz-yyyyyy.npy -> zz
        try:
            return  int(npy_filename.split('-')[0]),int(npy_filename.split('-')[1]), int(npy_filename.split('-')[2])
        except:
            return -1 # 或者抛出错误