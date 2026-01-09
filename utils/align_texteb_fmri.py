import scipy.io as scio
import interpdata
import json
import numpy as np
from nilearn.glm.first_level import hemodynamic_models
import utils.feature_convolve as feature_convolve
from data_utils import zscore

# 将文本特征的时间轴与fMRI对齐,layer选第6层
# 采用lanczos对齐
if __name__ == '__main__':
    texteb_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\embeddings\\bert\word-level'
    timesample_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\word-level'
    BOLD_num_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\preprocessed_data\\resp.json'
    save_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\embeddings\\bert\\word-level-align'
    with open(BOLD_num_path) as json_file:
        BOLD_num = json.load(json_file)

    for i in range(1,61):
        file_eb_path = texteb_path+'\story_'+str(i)+'_word_bert_1-12_768.mat'
        data = scio.loadmat(file_eb_path)['data'][6]
        file_ts_path = timesample_path + '\story_' + str(i) + '_word_time.mat'
        ts = scio.loadmat(file_ts_path)
        time_alingn = (ts['start'] + ts['end']) / 2

        BOLD_time_sample = np.arange(0, int(BOLD_num[str(i)]) * 0.71, 0.71)[:int(BOLD_num[str(i)])]
        temp = interpdata.lanczosinterp2D(data,time_alingn[0],BOLD_time_sample)
        temp2 = zscore(temp)

        file_eb_path_save = save_path + '\story_'+str(i)+'_word_bert_1-12_768.mat'
        scio.savemat(file_eb_path_save,{'data':temp})

# 采用HRF对齐
# if __name__ == '__main__':
#     texteb_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\embeddings\\bert\word-level'
#     timesample_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\word-level'
#     BOLD_num_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\preprocessed_data\\resp.json'
#     save_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\embeddings\\bert\\word-level-align-HRF'
#     with open(BOLD_num_path) as json_file:
#         BOLD_num = json.load(json_file)
#
#     for i in range(1, 61):
#         file_eb_path = texteb_path + '\story_' + str(i) + '_word_bert_1-12_768.mat'
#         data = scio.loadmat(file_eb_path)['data'][6]
#         file_ts_path = timesample_path + '\story_' + str(i) + '_word_time.mat'
#         ts = scio.loadmat(file_ts_path)
#         time_alingn = (ts['start'] + ts['end']) / 2
#
#         BOLD_time_sample = np.arange(0, int(BOLD_num[str(i)]) * 0.71, 0.71)[:int(BOLD_num[str(i)])]
#
#         hrf = hemodynamic_models.spm_hrf(0.71, oversampling=1)
#         temp = feature_convolve.convolve_downsample(data, time_alingn, hrf, 0, int(BOLD_num[str(i)]))
#         if temp.shape[0] != int(BOLD_num[str(i)]):
#             print('warning')
#             print(f'stroy{i}')
#
#         file_eb_path_save = save_path + '\story_' + str(i) + '_word_bert_1-12_768.mat'
#         scio.savemat(file_eb_path_save, {'data': temp})