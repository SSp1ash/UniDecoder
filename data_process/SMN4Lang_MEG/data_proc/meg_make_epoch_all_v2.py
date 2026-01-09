import mne
import numpy as np
import os
import scipy.io as scio
import torch
from scipy.stats import zscore

import config
import utils.interpdata
import einops

def SMN4Lang_badchannel(arr):
    indices = [8,318,319,320,321,322,323,324,325,326,327]
    indices_set = set(indices)

    remaining_indices = [i for i in range(arr.shape[0]) if i not in indices_set]
    return arr[remaining_indices]


def intercep_meg(data, sample, tmin=-0.25, tmax=0.25, freq_input=250):
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

def create_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    else:
        print(f"Directory {path} already exists.")

# 顺便concate一下
if __name__ == '__main__':

    start = 1
    end = 60
    # sub_list = ['03','04','05','06','07','08','09','10','11','12']
    # sub_list = ['02','09','10','11','12']
    # sub_list = ['08']
    sub_list = ['01','02','03','04','05','06','07','08','09','10','11','12']

    for sub_num in sub_list:
        path_meg_dir = config.rawdata_path + fr'/SMN4Lang/ds004078-download/derivatives/preprocessed_data/sub-{sub_num}/MEG/'
        # path_ts_dir = config.project_lfs_path + r'/SMN4Lang/dataset/bloom_word_times/'
        path_ts_dir = config.project_lfs_path + r'/SMN4Lang/dataset/bloom_word_times_dataclean/'
        save_dir = config.project_lfs_path + rf'/SMN4Lang/dataset/MEG/'
        create_directories(save_dir)

        concate_meg = []
        concate_rate = []
        concate_meg_noepoch = []
        for i in range(start, end+1):
            file_meg_name = rf'sub-{sub_num}_task-RDR_run-{i}_meg.fif'
            file_ts_name = rf'word_times_story{i}.npy'
            path_meg_data = os.path.join(path_meg_dir,file_meg_name)
            path_ts_data = os.path.join(path_ts_dir,file_ts_name)
            raw_data = mne.io.read_raw_fif(path_meg_data,preload=True)


            # 从1000hz降采样到200hz
            new_sampling_rate = 200
            raw_data.resample(new_sampling_rate)

            ts = np.load(path_ts_data)
            time_alingn = ts

            # 计算rate，每秒的char/word数量
            # times_second = int(raw_data.n_times / raw_data.info['sfreq'])
            # MEG_time_sample = np.arange(0, times_second * 1, 1)[:times_second]
            # rate = utils.interpdata.lanczosinterp2D(np.ones(time_alingn.shape[0]), time_alingn, MEG_time_sample)

            samples = np.round(time_alingn * new_sampling_rate).astype(int)

            data = SMN4Lang_badchannel(raw_data.get_data())
            data = zscore(data,1)
            # data_noepoch = einops.rearrange(data, 'b (c d) -> c b d', c=times_second,d=int(raw_data.info['sfreq']))

            x = intercep_meg(data,samples,-0.5,0.5,new_sampling_rate)
            x = x.astype(np.float32)
            concate_meg.append(x)
            # concate_rate.append(rate)
            # concate_meg_noepoch.append(data_noepoch)
        concate_meg = np.concatenate(concate_meg)

        torch.save(torch.from_numpy(concate_meg), save_dir + f'/meg_raw_sub{sub_num}_bloom1.1_v2.pth')

