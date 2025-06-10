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
import utils.base

@torch.no_grad
def extract_layer_embeddings(model, token_ids, split=None,layer=config.llama3_feature_layer):
    if split == None:
        eb = model(token_ids,output_hidden_states=True).hidden_states[layer]
    else:
        tokenid_list = torch.split(token_ids, split, dim=1)
        eb_list = [model(item,output_hidden_states=True).hidden_states[layer] for item in tokenid_list]
        eb = torch.cat(eb_list,dim=1)
    return eb

@torch.no_grad
def extract_layer_embeddings2(model, token_ids, split=None,layer=config.llama3_feature_layer):
    if split == None:
        eb = model(token_ids,output_hidden_states=True).hidden_states[layer]
    else:
        tokenid_list = torch.split(token_ids, split, dim=1)
        eb_list = [model(item,output_hidden_states=True).hidden_states[layer] for item in tokenid_list]
        eb = torch.cat(eb_list,dim=1)
    return eb


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

                # token_ids = torch.from_numpy(token_ids)

                utils.base.create_folder_if_not_exists(config.project_lfs_path + f'/Broderick2018/dataset/word_times_bloom_en')
                np.save(config.project_lfs_path + f'/Broderick2018/dataset/word_times_bloom_en/word_times_story{i}.npy', word_time)

                eb = extract_contextualized_embeddings(model, token_ids[None,:].to(device),layer,split)
                eb = eb.cpu().numpy()
                eb_list.append(eb)
            eb = np.concatenate(eb_list, axis=0)
            eb = torch.from_numpy(eb)

            torch.save(eb,config.project_lfs_path + f'/Broderick2018/dataset/eb_bloom1.1_{lan}_split{split}_{layer}layer.pth')

