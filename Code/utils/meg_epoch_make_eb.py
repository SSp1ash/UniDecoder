import mne
import numpy as np
import os
import scipy.io as scio
from utils.data_utils import zscore



if __name__ == '__main__':
    # Train set
    eb_path = '/media/test/data/gy/chinese_speech/annotations/embeddings/bert/word-level/'
    save_dir = '/media/test/data/gy/chinese_speech/MEG_process/MEG_dataset_eb'
    start = 1
    end = 57
    concate_eb = []
    for i in range(start, end+1):
        data = scio.loadmat(eb_path + f'story_{i}_word_bert_1-12_768.mat')
        x = data['data'][6]
        concate_eb.append(x)
    concate_eb = np.concatenate(concate_eb)
    savepath = os.path.join(save_dir,'train_eb.npz')
    np.savez_compressed(savepath, concate_eb)


    # Test set
    eb_path = '/media/test/data/gy/chinese_speech/annotations/embeddings/bert/word-level/'
    save_dir = '/media/test/data/gy/chinese_speech/MEG_process/MEG_dataset_eb'
    start = 58
    end = 60
    concate_eb = []
    for i in range(start, end+1):
        data = scio.loadmat(eb_path + f'story_{i}_word_bert_1-12_768.mat')
        x = data['data'][6]
        concate_eb.append(x)
    concate_eb = np.concatenate(concate_eb)
    savepath = os.path.join(save_dir,'test_eb.npz')
    np.savez_compressed(savepath, concate_eb)

