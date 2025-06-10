import nibabel as nib
import numpy as np
import scipy.io as scio
# from utils.data_utils import zscore
from scipy.stats import zscore
from utils.extract_roi import extract_speech
import config
from utils.interpdata import lanczosinterp2D
import os
import torch

roi = extract_speech(config.project_lfs_path + r'/SMN4Lang/CN001.aparc.a2009s.32k_fs_LR.dlabel.nii')



if __name__ == '__main__':
    start = 1
    end = 60
    TR = 0.71
    # sub_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
    sub_list = ['03','04','05','06','07','08','09','10','11','12']
    for sub in sub_list:
        # fmri_1d_path = f'/media/test/data/gy/chinese_speech/preprocessed_data/sub-{sub}/CIFTI'
        fmri_1d_path = config.rawdata_path + f'/SMN4Lang/ds004078-download/derivatives/preprocessed_data/sub-{sub}/CIFTI'
        data_list = []
        for i in range(start,end+1):
            file_path = fmri_1d_path + f'/sub-{sub}_task-RDR_run-{i}_bold.dtseries.nii'
            fMRI = nib.load(file_path).get_fdata()
            # fMRI_normal = zscore(fMRI)
            fMRI_normal = fMRI

            # 进行超分
            # time_data = scio.loadmat(f'/media/test/data/gy/chinese_speech/annotations/time_align/char-level/story_{i}_char_time.mat')
            # char_times = np.squeeze((time_data['start'] + time_data['end']) / 2)
            char_times = np.load(config.project_lfs_path + f'/SMN4Lang/dataset/bloom_word_times_dataclean/word_times_story{i}.npy')

            n = fMRI_normal.shape[0]
            TR_times = np.arange(0, 0 + n * TR, TR)[:n]
            fMRI_resample = lanczosinterp2D(fMRI_normal, TR_times, char_times)
            # fMRI_resample = fMRI_normal

            data_list.append(fMRI_resample)
            fmri_data = np.concatenate(data_list,axis=0)

        fmri_data = torch.from_numpy(fmri_data)[:,roi]

        torch.save(fmri_data, config.project_lfs_path + f'/SMN4Lang/dataset/fMRI/fMRI_sub{sub}_roi.pth')












