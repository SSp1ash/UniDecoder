import config
import numpy as np
import json
if __name__ == '__main__':
    lan = 'EN'
    with open(config.rawdata_path + fr'/LPPC-fMRI/ds003643-download/annotation/{lan}/TR.json', 'r',
              encoding='utf-8') as json_file:
        TR = json.load(json_file)

    i = 0

    TRs = np.array(list(TR.values()))
    for run, TR_num in TR.items():

        data_file = np.load(config.project_lfs_path + '/LPPC-fMRI/dataset/Dic_token_id_time.npz')
        word_time = data_file[f'segment_{i+1}_time_list']
        token_word = data_file[f'segment_{i+1}_token_list']
        token_ids = data_file[f'segment_{i+1}_token_id_list']
        i += 1
        np.save(config.project_lfs_path + f'/LPPC-fMRI/dataset/word_compare_gt/story{i}_EN.npy', token_word)
