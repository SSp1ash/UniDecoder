import mne
import numpy as np
import os
import scipy.io as scio
from utils.data_utils import zscore

def intercep_meg(data,sample,tmin=-0.5,tmax=0.5):
    freq = 1000
    batch_data = []
    for i in sample:
        batch_data.append(data[:,i+int(tmin*freq):i+int(tmax*freq)])
    return np.array(batch_data)


# 顺便concate一下
if __name__ == '__main__':
    start = 1
    end = 57
    path_meg_dir = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\preprocessed_data\sub-01\MEG'
    path_ts_dir = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\word-level'
    concate_meg = []

    for i in range(1, end+1):
        file_meg_name = f'sub-01_task-RDR_run-{i}_meg.fif'
        file_ts_name = f'story_{i}_word_time.mat'
        path_meg_data = os.path.join(path_meg_dir,file_meg_name)
        path_ts_data = os.path.join(path_ts_dir,file_ts_name)
        raw_data = mne.io.read_raw_fif(path_meg_data)
        ts = scio.loadmat(path_ts_data)
        time_alingn = np.unique((ts['start'] + ts['end'])/2)
        samples = np.round(time_alingn * 1000).astype(int)
        data = zscore(raw_data.get_data())
        x = intercep_meg(data,samples)
        concate_meg.append(x)
    concate_meg = np.concatenate(concate_meg,axis=0)
    np.savez_compressed('./train_meg_normal', concate_meg)

