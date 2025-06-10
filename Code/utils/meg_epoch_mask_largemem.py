import mne
import numpy as np
import os
import scipy.io as scio
from scipy.stats import zscore

def SMN4Lang_badchannel(arr):
    indices = [8,318,319,320,321,322,323,324,325,326,327]
    indices_set = set(indices)

    remaining_indices = [i for i in range(arr.shape[0]) if i not in indices_set]
    return arr[remaining_indices]


def intercep_meg(data,sample,tmin=-0.5,tmax=0.5):
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
    start = 1
    end = 57
    # sub_list = ['01']
    sub_list = ['02','03','04','05','06','07','08','09','10','11','12']
    for sub_num in sub_list:
        path_meg_dir = fr'/media/test/data/gy/chinese_speech/preprocessed_data/sub-{sub_num}/MEG'
        path_ts_dir = r'/media/test/data/gy/chinese_speech/annotations/time_align/word-level'
        save_dir = rf'/media/test/data/gy/chinese_speech/MEG_process/MEG_dataset_sub{sub_num}/'
        create_directories(save_dir)

        concate_meg = []
        for i in range(start, end+1):
            file_meg_name = rf'sub-{sub_num}_task-RDR_run-{i}_meg.fif'
            file_ts_name = rf'story_{i}_word_time.mat'
            path_meg_data = os.path.join(path_meg_dir,file_meg_name)
            path_ts_data = os.path.join(path_ts_dir,file_ts_name)
            raw_data = mne.io.read_raw_fif(path_meg_data,preload=True)

            new_sampling_rate = 250
            raw_data.resample(new_sampling_rate)

            ts = scio.loadmat(path_ts_data)
            time_alingn = ((ts['start'] + ts['end'])/2)[0]

            samples = np.round(time_alingn * new_sampling_rate).astype(int)

            data = SMN4Lang_badchannel(raw_data.get_data())
            data = zscore(data,1)

            x = intercep_meg(data,samples)
            concate_meg.append(x)
        concate_meg = np.concatenate(concate_meg)
        savepath = os.path.join(save_dir,'train_meg_zscore.npz')
        np.savez_compressed(savepath, concate_meg)


    start = 58
    end = 60
    sub_list = ['01']
    # sub_list = ['02','03','04','05','06','07','08','09','10','11','12']
    for sub_num in sub_list:
        path_meg_dir = fr'/media/test/data/gy/chinese_speech/preprocessed_data/sub-{sub_num}/MEG'
        path_ts_dir = r'/media/test/data/gy/chinese_speech/annotations/time_align/word-level'
        save_dir = rf'/media/test/data/gy/chinese_speech/MEG_process/MEG_dataset_sub{sub_num}/'
        create_directories(save_dir)

        concate_meg = []
        for i in range(start, end+1):
            file_meg_name = rf'sub-{sub_num}_task-RDR_run-{i}_meg.fif'
            file_ts_name = rf'story_{i}_word_time.mat'
            path_meg_data = os.path.join(path_meg_dir,file_meg_name)
            path_ts_data = os.path.join(path_ts_dir,file_ts_name)
            raw_data = mne.io.read_raw_fif(path_meg_data,preload=True)

            new_sampling_rate = 250
            raw_data.resample(new_sampling_rate)

            ts = scio.loadmat(path_ts_data)
            time_alingn = ((ts['start'] + ts['end'])/2)[0]

            samples = np.round(time_alingn * new_sampling_rate).astype(int)

            data = SMN4Lang_badchannel(raw_data.get_data())
            data = zscore(data,1)

            x = intercep_meg(data,samples)
            concate_meg.append(x)
        concate_meg = np.concatenate(concate_meg)
        savepath = os.path.join(save_dir,'test_meg_zscore.npz')
        np.savez_compressed(savepath, concate_meg)