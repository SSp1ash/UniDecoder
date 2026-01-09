# generate stimuli features
import scipy.io as scio
import numpy as np
from nilearn import glm
import os
zs = lambda v: (v-v.mean())/v.std()
from nilearn.glm.first_level import hemodynamic_models
import matplotlib.pyplot as plot

def convolve_downsample(data, word_time, hrf, blank_TR, ref_length):
    """
    convolves word embeddings with HRF and downsample to the sampling rating of fMRI
    inputs:
        in_file: embedding file
        time_file: time_align files which contain the onset and offset time of each word
        out_file: file to save convolved features
        hrf: spm hrf, used to convolve embeddings
        blank_TR: the number of blank TRs in the beginning of each run
        ref_length: the number of TRs of corresponding fMRI data, use this to cut the 
                    useless data in the end of the convolved features.
    """

    length = int(word_time[0][-1]*100) # upsample the temporal resolution to 0.01s
    time_series = np.zeros([length, data.shape[1]])
    t = 0
    # set the values of word offset time in upsampled time series as the word embeddings
    for j in range(length):
        if j == int(word_time[0][t]*100):
            time_series[j] = data[t]
            while(j == int(word_time[0][t]*100)):
                t += 1
                if t == data.shape[0]:
                    break
    conv_series = []
    # convolution
    for j in range(data.shape[1]):
        conv_series.append(np.convolve(hrf, time_series[:,j]))
    conv_series = np.stack(conv_series).T
    conv_series = conv_series[:length]
    # downsample
    conv_series_ds = [conv_series[j] for j in range(0, length, 65)]
    conv_series_ds = np.array(conv_series_ds)
        
    word_feature = zs(conv_series_ds[blank_TR:ref_length+blank_TR]) # z-score, dump the beginning blank TRs
    return word_feature.astype('float32')


if __name__ == '__main__':
    def generate_simulated_data():
        np.random.seed(0)
        num_words = 1000  # 词的数量
        embedding_dim = 768  # 词嵌入维度
        embeddings = np.random.rand(num_words, embedding_dim)  # 随机生成词嵌入
        times = np.cumsum(np.random.rand(num_words) + 1)  # 随机生成词出现的时间，递增

        # 保存数据
        scio.savemat('sim_embeddings.mat', {'data': embeddings})
        scio.savemat('sim_times.mat', {'end': times.reshape(1, -1)})

        return 'sim_embeddings.mat', 'sim_times.mat'

    in_file, time_file = generate_simulated_data()

    # 定义输出文件
    out_file = 'output.mat'

    # 定义HRF
    tr = 0.71  # TR时间
    hrf = hemodynamic_models.spm_hrf(tr,oversampling=1)
    # plot.plot(hrf)
    # plot.show()
    # hrf = glm.first_level.compute_regressor(np.exp(-np.arange(30) / tr), 'glover', frame_times=np.arange(0, 30, tr))

    # 运行convolve_downsample函数
    convolve_downsample(in_file, time_file, out_file, hrf, blank_TR=0, ref_length=641)