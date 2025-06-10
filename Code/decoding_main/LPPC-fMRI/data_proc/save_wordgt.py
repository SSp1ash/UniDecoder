import numpy as np
import pandas as pd
import json
import nibabel as nib
import torch
from scipy.stats import zscore
from transformers import BertTokenizer, GPT2LMHeadModel
from utils.interpdata import lanczosinterp2D
import utils.data_utils
from utils.split_word import split_words_and_times
import config
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llama3_fenci import llama3_fenci
from utils.extract_roi import extract_speech
import os
import re

def extract_contextualized_embeddings(model, input_tokens, layer_index=20, context_length=15):
    # 提取 token 的总数
    token_count = input_tokens.size(1)

    # 初始化新的嵌入存储结构
    contextualized_embeddings = []

    # 前 context_length 个 token 的嵌入直接取模型输出
    with torch.no_grad():
        # 仅处理前 context_length 个 token
        outputs = model(input_tokens[:, :context_length],output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embeddings = hidden_states[layer_index][0]  # 获取第一个样本的嵌入
        contextualized_embeddings.extend(embeddings)  # 直接加入结果

    # 对于 context_length 以后的 token，采用滑动窗口
    for i in range(context_length, token_count):
        # 取前 context_length 个 token 作为滑动窗口
        window_start = i - context_length
        window_end = i + 1

        # 构建窗口的输入
        window_input_tokens = input_tokens[:, window_start:window_end]

        # 使用模型计算这个窗口的嵌入
        with torch.no_grad():
            window_outputs = model(window_input_tokens,output_hidden_states=True)
            window_hidden_states = window_outputs.hidden_states

        # 取窗口中最后一个 token 的嵌入
        last_token_embedding = window_hidden_states[layer_index][0, -1]
        contextualized_embeddings.append(last_token_embedding)

    # 将新的嵌入 tensor 化
    contextualized_embeddings = torch.stack(contextualized_embeddings)

    return contextualized_embeddings

if __name__ == '__main__':
    with torch.no_grad():
        # 设置映射到大模型哪一层
        layer = 20
        sub_num = 'CN001'

        roi = extract_speech(config.project_lfs_path + r'/LPPC-fMRI/CN001.aparc.a2009s.32k_fs_LR.dlabel.nii')
        vox = roi.__len__()

        # lan_list = ['CN','EN','FR']
        device = 'cuda:7'
        # model_id = "/home/guoyi/llm_model/bloom3b-3b"
        model_id = "/home/guoyi/llm_model/bloom-1b1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            # torch_dtype=dtype,
        )
        split = 15

        fMRI_list = sorted([f for f in os.listdir(config.rawdata_path + f'/LPPC-fMRI/ds003643-download/fMRI_data/sub-{sub_num}') if re.search(r'run-(\d+)', f)],
               key=lambda x: int(re.search(r'run-(\d+)', x).group(1)))

        fMRI_list = [config.rawdata_path + f'/LPPC-fMRI/ds003643-download/fMRI_data/sub-{sub_num}/' + i for i in fMRI_list]

        lan_list = ['CN']
        for lan in lan_list:
            data_info_all = pd.read_csv(config.rawdata_path + fr'/LPPC-fMRI/ds003643-download/annotation/{lan}/lpp{lan}_word_information.csv')
            emb_train_list = []
            fmri_resample_train_list = []
            fmri_train_list = []
            wr_train_list = []
            eb_downsample_list = []
            with open(config.rawdata_path + fr'/LPPC-fMRI/ds003643-download/annotation/{lan}/TR.json', 'r', encoding='utf-8') as json_file:
                TR = json.load(json_file)

            i = 0

            TRs = np.array(list(TR.values()))
            for run, TR_num in TR.items():
                run_no = int(run[-1])

                ##########
                # 指定run_no
                # run_no = 9
                ###########

                index = (data_info_all['section'] == run_no)
                data_info = data_info_all[index]
                time_alingn = np.array((data_info['onset'] + data_info['offset']) / 2)

                # 改为字符级别的编码
                words = np.array(data_info['word'])
                chars, char_time = split_words_and_times(words, time_alingn)
                char_time = char_time[np.newaxis,:]
                text = ''.join(list(chars))
                token_ids, tokens, offset_mapping = llama3_fenci(tokenizer,text)
                i += 1
                np.save(config.project_lfs_path + f'/LPPC-fMRI/dataset/word_compare_gt/story{i}_CN.npy',tokens)
