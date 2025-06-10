import numpy as np
import torch
import config
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    with torch.no_grad():

        # 设置映射到大模型哪一层
        layer = 20
        device = 'cuda:1'
        model_id = "/home/guoyi/llm_model/bloom-1b1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            # torch_dtype=dtype,
        )
        split = 15

        lan_list = ['EN']
        for lan in lan_list:
            data_file = np.load(config.project_lfs_path + '/Broderick2018/dataset/Dic_token_id_time_Broderick.npz',allow_pickle=True)
            eb_list = []

            start = 1
            end = 20
            for i in range(start,end+1):
                data = data_file[str(i)].item()

                word_time = data['time_list']
                token_word = data['token_list']
                token_ids = data['token_id_list']
                np.save(config.project_lfs_path + f'/Broderick2018/dataset/word_compare_gt/story{i}_EN.npy',token_word)
