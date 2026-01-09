import config
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
def read_txt_to_str(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

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
    device = 'cuda:3'
    model_id = "/home/guoyi/llm_model/bloom-1b1"
    # dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        # torch_dtype=dtype,
    )

    path = config.rawdata_path + '/LPPC-fMRI/ds003643-download/annotation/EN/EN2NL.txt'
    # 使用示例
    file_content = read_txt_to_str(path)
    tokens = tokenizer.encode(file_content)[:300]
    tokens = np.array(tokens)
    token_ids = torch.from_numpy(tokens)

    eb = extract_contextualized_embeddings(model, token_ids[None, :].to(device), 20, 15)
    eb = eb.cpu()
    torch.save(eb,config.project_lfs_path + '/LPPC-fMRI/dataset/fMRI/eb_bloom1.1_NL_split15_20layer.pth')

    print(123)