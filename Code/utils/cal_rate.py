import numpy as np
import scipy.io as scio
import json
import interpdata

if __name__ == '__main__':

    timesample_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\char-level'
    BOLD_num_path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\preprocessed_data\\resp.json'
    with open(BOLD_num_path) as json_file:
        BOLD_num = json.load(json_file)
    rates = []
    # start = 1;end = 57
    start = 58;end=60


    for i in range(start, end+1):
        file_ts_path = timesample_path + '\story_' + str(i) + '_char_time.mat'
        ts = scio.loadmat(file_ts_path)
        time_alingn = (ts['start'] + ts['end']) / 2
        BOLD_time_sample = np.arange(0, int(BOLD_num[str(i)]) * 0.71, 0.71)[:int(BOLD_num[str(i)])]
        rates.append(interpdata.lanczosinterp2D(np.ones(time_alingn.shape[1]), time_alingn[0], BOLD_time_sample))
    nz_rate = np.concatenate(rates,axis=0)
    nz_rate = np.nan_to_num(nz_rate).reshape([-1, 1])
    mean_rate = np.mean(nz_rate)
    rate = nz_rate - mean_rate
    # scio.savemat('./train_WR.mat',{'rate':rate,"mean_rate":mean_rate})
    scio.savemat('./test_WR.mat',{'rate':nz_rate})



