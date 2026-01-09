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


def compress_channels_with_averaging(eeg_data, target_channels=58):
    """
    使用平均方法将EEG数据从128通道压缩到目标通道数(如58通道)

    参数:
        eeg_data: numpy数组，形状为 [128, samples]
        target_channels: 目标通道数，默认为58

    返回:
        compressed_data: 压缩后的EEG数据，形状为 [target_channels, samples]
    """
    n_channels, n_samples = eeg_data.shape

    if n_channels != 128:
        print(f"警告：输入数据有{n_channels}个通道，不是128个")

    # 确定每个输出通道应该平均多少输入通道
    channels_per_group = n_channels / target_channels

    # 初始化输出数组
    compressed_data = np.zeros((target_channels, n_samples))

    for i in range(target_channels):
        # 计算当前输出通道对应的输入通道范围
        start_idx = int(i * channels_per_group)
        end_idx = int((i + 1) * channels_per_group)

        # 确保end_idx不超过原始通道数
        end_idx = min(end_idx, n_channels)

        # 对该范围内的通道取平均
        if start_idx < end_idx:
            compressed_data[i] = np.mean(eeg_data[start_idx:end_idx], axis=0)

    return compressed_data


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
    from models.EEG_feature_extra_EEGPT import *
    model = FeatureExtra()
    device = 'cuda:1'
    model = model.to(device)

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



            # 从1000hz降采样到640hz
            new_sampling_rate = 640
            raw_data.resample(new_sampling_rate)

            ts = np.load(path_ts_data)
            time_alingn = ts

            # 计算rate，每秒的char/word数量
            times_second = int(raw_data.n_times / raw_data.info['sfreq'])
            eeg_time_sample = np.arange(0, times_second * 1, 1)[:times_second]
            rate = utils.interpdata.lanczosinterp2D(np.ones(time_alingn.shape[0]), time_alingn, eeg_time_sample)

            samples = np.round(time_alingn * new_sampling_rate).astype(int)

            data = Broderick2018_badchannel(raw_data.get_data())
            data = compress_channels_with_averaging(data)
            data = zscore(data,1)
            # data_noepoch = einops.rearrange(data, 'b (c d) -> c b d', c=times_second,d=int(raw_data.info['sfreq']))

            x = intercep_eeg(data,samples,-0.8, 0.8, 640)
            x = torch.from_numpy(x).float().to(device)
            y = model(x)
            concate_eeg.append(y)

        concate_eeg = torch.concatenate(concate_eeg)

        torch.save(concate_eeg, save_dir + f'/eeg_raw_sub{sub_num}_v4_bloom1.1.pth')

