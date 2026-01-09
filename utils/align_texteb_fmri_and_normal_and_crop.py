import scipy.io as scio
import interpdata
import json
import numpy as np
import utils.feature_convolve as feature_convolve
# from data_utils import zscore
from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler
import nibabel
import os

# 将文本特征的时间轴与fMRI对齐,layer选第6层
# 采用lanczos对齐
if __name__ == '__main__':
    # sub_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
    sub_list = ['01']
    for sub_num in sub_list:
        start = 1
        end = 57

        base_path = r'/media/test/data/gy/chinese_speech'

        texteb_path = os.path.join(base_path,r'annotations/embeddings/bert/word-level')
        timesample_path = os.path.join(base_path,r'annotations/time_align/word-level')
        BOLD_num_path = os.path.join(base_path,r'preprocessed_data/resp.json')
        with open(BOLD_num_path) as json_file:
            BOLD_num = json.load(json_file)
        temp_list = []
        data_list = []
        for i in range(start,end+1):
            BOLD_data_path = os.path.join(base_path,fr'preprocessed_data/sub-{sub_num}/CIFTI/sub-{sub_num}_task-RDR_run-{i}_bold.dtseries.nii')

            file_eb_path = texteb_path+'/story_'+str(i)+'_word_bert_1-12_768.mat'
            data = scio.loadmat(file_eb_path)['data'][6]
            file_ts_path = timesample_path + '/story_' + str(i) + '_word_time.mat'
            ts = scio.loadmat(file_ts_path)
            time_alingn = (ts['start'] + ts['end']) / 2
            BOLD_time_sample = np.arange(0, int(BOLD_num[str(i)]) * 0.71, 0.71)[:int(BOLD_num[str(i)])]
            temp = interpdata.lanczosinterp2D(data,time_alingn[0],BOLD_time_sample)
            a = np.nonzero(temp[:,0])[0][0]
            b = np.nonzero(temp[:,0])[0][-1]
            temp2 = temp[a:b+1,:]
            # temp2 = temp2 / temp2.max(1)[:,None]
            # temp2 = temp2 / temp2.max()

            temp_list.append(temp2)
            # 顺便把fMRI也裁剪
            fmri_data = nibabel.load(BOLD_data_path).get_fdata()[a:b+1]
            # data_list.append(fmri_data / fmri_data.max())
            data_list.append(fmri_data)

        texteb_normal = np.concatenate(temp_list)
        data_normal = np.concatenate(data_list)
        data_normal = zscore(data_normal)

        # np.savez_compressed('../dataset/SMN4Lang/train_texteb_data', texteb_normal)
        # np.savez_compressed(f'../dataset/SMN4Lang/train_fmri_1Ddata_guiyi-sub{sub_num}', data_normal)

        np.savez_compressed('../dataset/SMN4Lang/train_texteb_data', texteb_normal)
        np.savez_compressed(f'../dataset/SMN4Lang/train_fmri_1Ddata_zscore-sub{sub_num}', data_normal)

        #for test -----------------------------------------------------------------------------------------------------------------
        start = 58
        end = 60

        base_path = r'/media/test/data/gy/chinese_speech'

        texteb_path = os.path.join(base_path,r'annotations/embeddings/bert/word-level')
        timesample_path = os.path.join(base_path,r'annotations/time_align/word-level')
        BOLD_num_path = os.path.join(base_path,r'preprocessed_data/resp.json')
        with open(BOLD_num_path) as json_file:
            BOLD_num = json.load(json_file)
        temp_list = []
        data_list = []
        for i in range(start,end+1):
            BOLD_data_path = os.path.join(base_path,fr'preprocessed_data/sub-{sub_num}/CIFTI/sub-{sub_num}_task-RDR_run-{i}_bold.dtseries.nii')

            file_eb_path = texteb_path+'/story_'+str(i)+'_word_bert_1-12_768.mat'
            data = scio.loadmat(file_eb_path)['data'][6]
            file_ts_path = timesample_path + '/story_' + str(i) + '_word_time.mat'
            ts = scio.loadmat(file_ts_path)
            time_alingn = (ts['start'] + ts['end']) / 2
            BOLD_time_sample = np.arange(0, int(BOLD_num[str(i)]) * 0.71, 0.71)[:int(BOLD_num[str(i)])]
            temp = interpdata.lanczosinterp2D(data,time_alingn[0],BOLD_time_sample)
            a = np.nonzero(temp[:,0])[0][0]
            b = np.nonzero(temp[:,0])[0][-1]
            temp2 = temp[a:b + 1, :]
            # temp2 = temp2 / temp2.max(1)[:, None]
            # temp2 = temp2 / temp2.max()

            temp_list.append(temp2)
            # 顺便把fMRI也裁剪
            fmri_data = nibabel.load(BOLD_data_path).get_fdata()[a:b + 1]
            fmri_data = zscore(fmri_data)
            # data_list.append(fmri_data / fmri_data.max())
            data_list.append(fmri_data)

        texteb_normal = np.concatenate(temp_list)
        data_normal = np.concatenate(data_list)
        data_normal = zscore(data_normal)
        # np.savez_compressed('../dataset/SMN4Lang/test_texteb_data', texteb_normal)
        # np.savez_compressed(f'../dataset/SMN4Lang/test_fmri_1Ddata_guiyi-sub{sub_num}', data_normal)

        np.savez_compressed('../dataset/SMN4Lang/test_texteb_data', texteb_normal)
        np.savez_compressed(f'../dataset/SMN4Lang/test_fmri_1Ddata_zscore-sub{sub_num}', data_normal)