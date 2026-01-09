import textgrid
import numpy as np
import config
def read_textgrid(file_path):
    # 读取TextGrid文件
    tg = textgrid.TextGrid.fromFile(file_path)

    # 假设文件中包含一个名为 "words" 的层次
    word_tiers = tg.getList('words')  # 根据实际层名调整

    words = []
    times = []

    # 定义需要过滤的无意义标记
    unwanted_marks = {'sil', 'sp', ''}

    # 遍历层次，提取有效的单词及其对应的时间轴
    for tier in word_tiers:
        for interval in tier:
            word = interval.mark
            if word and word.lower() not in unwanted_marks:
                words.append(word)  # 添加有效的词语
                times.append((interval.minTime, interval.maxTime))  # 对应的时间轴

    return words, times

if __name__ == '__main__':
    # file_path = "/home/guoyi/download_dataset/Broderick2018/ds004408-download/stimuli/"
    # storys = {}
    # start = 1
    # end = 20
    # for i in range(start,end+1):
    #     words, times = read_textgrid(file_path + f'audio{str(i).zfill(2)}.TextGrid')
    #     # 转为小写
    #     words = [word.lower() for word in words]
    #     storys[str(i)] = (words,times)
    # np.save("./Broderick_storys.npy", storys, allow_pickle=True)

    # loaded_data = np.load("Broderick_storys.npy", allow_pickle=True).item()

    start = 1
    end = 20
    for i in range(start,end+1):
        data_all = np.load(config.project_lfs_path + '/Broderick2018/dataset/Dic_token_id_time_Broderick.npz',allow_pickle=True)
        data = data_all[str(i)].item()

        print(123)