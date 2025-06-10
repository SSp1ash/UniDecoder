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


def intercep_meg(data,sample,tmin=-0.25,tmax=0.25):
    freq = 250
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
    # -----------------Train--------------------
    start = 1
    end = 57
    # sub_list = ['01']
    sub_list = ['02','03','04','05','06','07','08','09','10','11','12']
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


            # 从1000hz降采样到250hz
            new_sampling_rate = 250
            raw_data.resample(new_sampling_rate)

            ts = np.load(path_ts_data)
            time_alingn = ts

            # 计算rate，每秒的char/word数量
            times_second = int(raw_data.n_times / raw_data.info['sfreq'])
            MEG_time_sample = np.arange(0, times_second * 1, 1)[:times_second]
            rate = utils.interpdata.lanczosinterp2D(np.ones(time_alingn.shape[0]), time_alingn, MEG_time_sample)

            samples = np.round(time_alingn * new_sampling_rate).astype(int)

            data = SMN4Lang_badchannel(raw_data.get_data())
            data = zscore(data,1)
            data_noepoch = einops.rearrange(data, 'b (c d) -> c b d', c=times_second,d=int(raw_data.info['sfreq']))

            x = intercep_meg(data,samples)
            concate_meg.append(x)
            concate_rate.append(rate)
            concate_meg_noepoch.append(data_noepoch)
        concate_meg = np.concatenate(concate_meg)
        # concate_meg_noepoch = np.concatenate(concate_meg_noepoch)
        # concate_rate = np.concatenate(concate_rate)

        savepath = os.path.join(save_dir,'train_meg_zscore.npz')
        savepath_noepoch = os.path.join(save_dir,'train_meg_zscore_noepoch.npz')
        savepath_rate = os.path.join(save_dir, 'train_WR.npz')


        # np.savez_compressed(savepath, concate_meg)
        torch.save(torch.from_numpy(concate_meg),f'/home/guoyi/train_meg_normal_sub{sub_num}_bloom.pth')

        # np.savez_compressed(savepath_noepoch, concate_meg_noepoch)
        # np.savez_compressed(savepath_rate, concate_rate)


    # -------------------Test----------------------
    start = 58
    end = 60
    sub_list = ['01']
    # sub_list = ['02','03','04','05','06','07','08','09','10','11','12']
    for sub_num in sub_list:
        path_meg_dir = config.rawdata_path + fr'/SMN4Lang/ds004078-download/derivatives/preprocessed_data/sub-{sub_num}/MEG/'
        path_ts_dir = config.project_lfs_path + r'/SMN4Lang/dataset/bloom_word_times/'
        save_dir = config.project_lfs_path + rf'/SMN4Lang/dataset/MEG/'
        # create_directories(save_dir)

        concate_meg = []
        concate_rate = []
        concate_meg_noepoch = []
        for i in range(start, end + 1):
            file_meg_name = rf'sub-{sub_num}_task-RDR_run-{i}_meg.fif'
            file_ts_name = rf'word_times_story{i}.npy'
            path_meg_data = os.path.join(path_meg_dir, file_meg_name)
            path_ts_data = os.path.join(path_ts_dir, file_ts_name)
            raw_data = mne.io.read_raw_fif(path_meg_data, preload=True)

            new_sampling_rate = 250
            raw_data.resample(new_sampling_rate)

            ts = np.load(path_ts_data)
            time_alingn = ts

            # 计算rate，每秒的char/word数量
            times_second = int(raw_data.n_times / raw_data.info['sfreq'])
            MEG_time_sample = np.arange(0, times_second * 1, 1)[:times_second]
            rate = utils.interpdata.lanczosinterp2D(np.ones(time_alingn.shape[0]), time_alingn, MEG_time_sample)

            samples = np.round(time_alingn * new_sampling_rate).astype(int)

            data = SMN4Lang_badchannel(raw_data.get_data())
            data = zscore(data, 1)
            data_noepoch = einops.rearrange(data, 'b (c d) -> c b d', c=times_second, d=int(raw_data.info['sfreq']))

            x = intercep_meg(data, samples)
            concate_meg.append(x)
            concate_rate.append(rate)
            concate_meg_noepoch.append(data_noepoch)
        concate_meg = np.concatenate(concate_meg)
        # concate_meg_noepoch = np.concatenate(concate_meg_noepoch)
        # concate_rate = np.concatenate(concate_rate)

        savepath = os.path.join(save_dir, 'train_meg_zscore.npz')
        savepath_noepoch = os.path.join(save_dir, 'train_meg_zscore_noepoch.npz')
        savepath_rate = os.path.join(save_dir, 'train_WR.npz')

        # np.savez_compressed(savepath, concate_meg)
        torch.save(torch.from_numpy(concate_meg), f'/home/guoyi/test_meg_normal_sub{sub_num}_bloom.pth')

        # np.savez_compressed(savepath_noepoch, concate_meg_noepoch)
        # np.savez_compressed(savepath_rate, concate_rate)