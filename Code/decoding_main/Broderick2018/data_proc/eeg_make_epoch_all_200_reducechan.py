import mne
import numpy as np
import os
import scipy.io as scio
import torch
from scipy.stats import zscore

import config
import utils.interpdata
import einops
from mne.channels import make_standard_montage

from sklearn.decomposition import PCA
import numpy as np


def compress_channels_with_pca(eeg_data, target_channels=58):
    """
    使用PCA将EEG数据从128通道压缩到目标通道数(如58通道)

    参数:
        eeg_data: numpy数组，形状为 [128, samples] 或 [samples, 128]
        target_channels: 目标通道数，默认为58

    返回:
        compressed_data: 压缩后的EEG数据，形状为 [target_channels, samples] 或 [samples, target_channels]
    """
    # 确定输入数据的格式
    # 如果通道在第二维，需要转置以使通道在第一维
    if eeg_data.shape[0] > eeg_data.shape[1]:  # 假设样本数大于通道数
        data_for_pca = eeg_data.T  # [128, samples]
        transposed = False
    else:
        data_for_pca = eeg_data  # 已经是 [128, samples]
        transposed = True

    # 初始化PCA
    pca = PCA(n_components=target_channels)

    # 应用PCA变换
    # PCA期望输入形状为 [samples, features]，所以需要转置
    transformed_data = pca.fit_transform(data_for_pca.T).T  # 结果为 [target_channels, samples]

    # 如果原始输入是 [samples, 128]，则返回 [samples, target_channels]
    if not transposed:
        transformed_data = transformed_data.T

    print(f"解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")

    return transformed_data

def Broderick2018_badchannel(arr):
    # 该数据集 34,84,85,88,89 通道bad
    # indices = [33,83,54,87,88]
    indices = [-1]
    indices_set = set(indices)

    remaining_indices = [i for i in range(arr.shape[0]) if i not in indices_set]
    return arr[remaining_indices]


def intercep_eeg(data, sample, tmin=-0.25, tmax=0.25, freq_input=250):
    freq = freq_input
    batch_data = []
    time_length = data.shape[1]

    for i in sample:
        start_idx = i + int(tmin * freq)
        end_idx = i + int(tmax * freq)

        # 检查是否超出边界
        if start_idx >= 0 and end_idx <= time_length:
            # 正常情况，直接添加数据
            batch_data.append(data[:, start_idx:end_idx])
        else:
            # 创建填充数组
            segment_length = end_idx - start_idx
            padded_segment = np.zeros((data.shape[0], segment_length))

            # 计算有效数据的范围
            valid_start = max(0, start_idx)
            valid_end = min(time_length, end_idx)

            # 填充有效数据
            if valid_start < valid_end:
                offset = valid_start - start_idx
                padded_segment[:, offset:offset + (valid_end - valid_start)] = data[:, valid_start:valid_end]

            batch_data.append(padded_segment)

    return np.array(batch_data)

# def intercep_eeg(data,sample,tmin=-0.25,tmax=0.25,freq_input=250):
#     freq = freq_input
#     batch_data = []
#     for i in sample:
#         batch_data.append(data[:,i+int(tmin*freq):i+int(tmax*freq)])
#     return np.array(batch_data)

def create_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists.")


from models.EEG_MLP_multisub_EEGPT import BrainEncoder

# 顺便concate一下
if __name__ == '__main__':

    start = 1
    end = 20
    sub_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019']
    for sub_num in sub_list:
        path_eeg_dir = config.rawdata_path + fr'/Broderick2018/ds004408-download/sub-{sub_num}/eeg/'
        path_ts_dir = config.project_lfs_path + r'/Broderick2018/dataset/word_times_bloom_en/'
        save_dir = config.project_lfs_path + rf'/Broderick2018/dataset/EEG/'
        create_directories(save_dir)

        concate_eeg = []

        concate_eeg_noepoch = []
        for i in range(start, end+1):
            file_eeg_name = rf'sub-{sub_num}_task-listening_run-{str(i).zfill(2)}_eeg.vhdr'
            file_ts_name = rf'word_times_story{i}.npy'
            path_eeg_data = os.path.join(path_eeg_dir,file_eeg_name)
            path_ts_data = os.path.join(path_ts_dir,file_ts_name)
            raw_data = mne.io.read_raw_brainvision(path_eeg_data,preload=True)

            new_sampling_rate = 200
            raw_data.resample(new_sampling_rate)

            ts = np.load(path_ts_data)
            time_alingn = ts

            # 计算rate，每秒的char/word数量
            times_second = int(raw_data.n_times / raw_data.info['sfreq'])
            eeg_time_sample = np.arange(0, times_second * 1, 1)[:times_second]
            rate = utils.interpdata.lanczosinterp2D(np.ones(time_alingn.shape[0]), time_alingn, eeg_time_sample)

            samples = np.round(time_alingn * new_sampling_rate).astype(int)

            data = Broderick2018_badchannel(raw_data.get_data())

            data = compress_channels_with_pca(data,64)

            data = zscore(data,1)
            # data_noepoch = einops.rearrange(data, 'b (c d) -> c b d', c=times_second,d=int(raw_data.info['sfreq']))

            x = intercep_eeg(data,samples,-0.5, 0.5, 200)
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            concate_eeg.append(x)
        concate_eeg = torch.concatenate(concate_eeg)
        torch.save(concate_eeg, save_dir + f'/eeg_raw_sub{sub_num}_200_64_20250404_bloom1.1.pth')

