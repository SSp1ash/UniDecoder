import mne
import numpy as np
import os
import scipy.io as scio
import torch
from scipy.stats import zscore

import config
import utils.interpdata
import einops

def Broderick2018_badchannel(arr):
    # 该数据集 34,84,85,88,89 通道bad
    # indices = [33,83,54,87,88]
    indices = [-1]
    indices_set = set(indices)

    remaining_indices = [i for i in range(arr.shape[0]) if i not in indices_set]
    return arr[remaining_indices]


def intercep_eeg(data,sample,tmin=-0.25,tmax=0.25,freq_input=250):
    freq = freq_input
    batch_data = []
    for i in sample:
        batch_data.append(data[:,i+int(tmin*freq):i+int(tmax*freq)])
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
    end = 20
    sub_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019']
    for sub_num in sub_list:
        path_eeg_dir = config.rawdata_path + fr'/Broderick2018/ds004408-download/sub-{sub_num}/eeg/'
        path_ts_dir = config.project_lfs_path + r'/Broderick2018/dataset/word_times_bloom_en/'
        save_dir = config.project_lfs_path + rf'/Broderick2018/dataset/EEG/'
        create_directories(save_dir)

        concate_eeg = []
        concate_rate = []
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
            data = zscore(data,1)
            # data_noepoch = einops.rearrange(data, 'b (c d) -> c b d', c=times_second,d=int(raw_data.info['sfreq']))

            x = intercep_eeg(data,samples,-0.5,0.5,200)
            concate_eeg.append(x)
            concate_rate.append(rate)
            # concate_eeg_noepoch.append(data_noepoch)
        concate_eeg = np.concatenate(concate_eeg)

        torch.save(torch.from_numpy(concate_eeg), save_dir + f'/eeg_raw_sub{sub_num}_v2_bloom1.1.pth')

