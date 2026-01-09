import torch
import numpy as np
import scipy.io as scio
import config
from transformers import BertTokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llama3_fenci import llama3_fenci
import os

# 这个版本是去掉了除了逗号和句号之外的数据


def filter_characters_and_time_axis(characters: np.ndarray, time_axis: np.ndarray):
    # 将字符数组转换为字符串
    char_str = ''.join(characters)

    # 只保留字母、数字、逗号和句号
    filtered_chars = [c for c in char_str if c.isalnum() or c in ('，', '。')]

    # 找出保留字符的索引
    indices = [i for i, c in enumerate(char_str) if c.isalnum() or c in ('，', '。')]

    # 根据索引过滤时间轴
    filtered_time_axis = time_axis[:, indices]

    return np.array(filtered_chars), filtered_time_axis

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹已创建：{folder_path}")
    else:
        print(f"文件夹已存在：{folder_path}")


@torch.no_grad
def extract_layer_embeddings(model, token_ids, split=None):
    if split == None:
        eb = model(token_ids,output_hidden_states=True).hidden_states[config.llama3_feature_layer]
    else:
        tokenid_list = torch.split(token_ids, split, dim=1)
        eb_list = [model(item,output_hidden_states=True).hidden_states[config.llama3_feature_layer] for item in tokenid_list]
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
    device = 'cuda:4'
    # device = 'cpu'
    model_id = "/home/guoyi/llm_model/bloom-1b1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        # torch_dtype=dtype,
    )
    # 设置嵌入的上下文长度
    split = 15
    if split > 0:
        create_folder_if_not_exists(config.rawdata_path + f'/SMN4Lang/ds004078-download/derivatives/annotations/embeddings/bloom1.1b/word-level_split{split}')

    char_path = config.rawdata_path + '/SMN4Lang/ds004078-download/derivatives/annotations/time_align/char-level/'

    start = 1
    end = 60
    ebs = []

    for i in range(start, end + 1):
        data = scio.loadmat(char_path + f'story_{i}_char_time.mat')
        # 为了简化，防止编码的token增多，只取一个字符
        char = data['char'].astype(dtype='U1')
        # 字符时间轴划分为词语级别
        char_time = (data['start'] + data['end']) / 2

        char, char_time = filter_characters_and_time_axis(char,char_time)

        text = ''.join(list(char))
        token_ids, tokens, offset_mapping = llama3_fenci(tokenizer, text)

        word_time = [char_time[0][item[0]:item[1]].mean() for item in offset_mapping]
        np.save(config.project_lfs_path + f'/SMN4Lang/dataset/bloom_word_times_dataclean/word_times_story{i}.npy', word_time)
        token_ids = token_ids.to(device)[None, :]
        eb = extract_contextualized_embeddings(model, token_ids, layer_index=20)
        if split == 0:
            torch.save(eb,config.rawdata_path + f'/SMN4Lang/ds004078-download/derivatives/annotations/embeddings/bloom_dataclean/word-level/story_{i}_bloom1.1_{config.bloom1b1_feature_layer}.pth')
        else:
            torch.save(eb,config.rawdata_path + f'/SMN4Lang/ds004078-download/derivatives/annotations/embeddings/bloom_dataclean/word-level_split{split}/story_{i}_bloom1.1_{config.bloom1b1_feature_layer}.pth')
        ebs.append(eb)
    train_ebs = torch.cat(ebs,dim=0)
    if split > 0:
        torch.save(train_ebs,config.project_lfs_path + f'/SMN4Lang/dataset/fMRI/eb_bloom_dataclean_di{config.bloom1b1_feature_layer}layer_split{split}.pth')
    else:
        torch.save(train_ebs,config.project_lfs_path + f'/SMN4Lang/dataset/fMRI/eb_bloom_dataclean_di{config.bloom1b1_feature_layer}layer.pth')



