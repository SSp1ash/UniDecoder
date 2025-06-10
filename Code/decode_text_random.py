import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from llm_decode_text_lib.Decoder import Decoder, Hypothesis
from llm_decode_text_lib.LM import LanguageModel
from llm_decode_text_lib.StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from llm_decode_text_lib.llama3_chinese import LLaMA3
import scipy.io as scio
from transformers import BertTokenizer, GPT2LMHeadModel
from transformers import AlbertTokenizer, BertModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import warnings
import time

# 忽略与广播相关的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size .* different to the input size .*")

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

@torch.no_grad
def gen_eb_llama3(text,model,tokenizer,feature_layer = config.llama3_feature_layer):
    input = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(device)
    out = model(input, output_hidden_states=True)['hidden_states'][feature_layer]
    return out

@torch.no_grad
def gen_eb_llama3_char(text,model,tokenizer):
    # input = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(device)

    input = [tokenizer.encode(item, return_tensors='pt', padding=False, add_special_tokens=False) for item in list(text)]
    input = torch.cat(input).T.to(device)

    out = model(input, output_hidden_states=True)['hidden_states'][config.llama3_feature_layer]
    return out

@torch.no_grad
def gen_eb_llama3_split(text,model,tokenizer):
    token = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(device)
    out = extract_contextualized_embeddings(model,token)
    out = out[None,:]
    return out


@torch.no_grad
def contra_eb_char_level(extend_embs, gt, indexs, device):
    criterion = nn.MSELoss(reduction='none')

    extend_embs = np.stack(extend_embs)
    extend_embs = torch.from_numpy(extend_embs).to(device)
    cos_sim = F.cosine_similarity(extend_embs,gt[:,indexs])
    mse = criterion(extend_embs,gt[:,indexs]).sum(1)


    if mse.shape[0] == 1:
        mse_like = mse / mse
    else:
        mse_like = 1 - (mse - mse.min()) / (mse.max() - mse.min())
        # mse_like = 1 - (mse/mse.max())
    like = 0.2 * mse_like + 0.8 * cos_sim
    # like = cos_sim

    return like

@torch.no_grad
def gen_word_level_llama3(referen_eb, model, tokenizer, device, Lang, feature_layer = config.llama3_feature_layer):
    nuc_max = 0

    length = referen_eb.shape[1]
    llama3 = LLaMA3(model,tokenizer, device)
    # contrxt_words 提取特征的时候上下文长度
    features = LMFeatures(model=llama3, tokenizer=tokenizer, layer=feature_layer, context_words=15)
    # lm = LanguageModel(llama3, nuc_mass=0.9, nuc_ratio=0.03, output_lan=Lang)
    lm = LanguageModel(llama3, nuc_mass=0.9, nuc_ratio=0.03, output_lan=Lang)

    word_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    decoder = Decoder(word_times, 15)
    for sample_index in range(length):
        print(sample_index)
        # ncontext 生成文本的时候用多少上下文预测
        ncontext = 15
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        # beam_nucs = lm.beam_propose_avoidnull(decoder.beam, ncontext)

        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if nuc.__len__() > nuc_max:
                nuc_max = nuc.__len__()

            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))

            likelihoods = contra_eb_char_level(extend_embs, referen_eb, sample_index, device)
            likelihoods = likelihoods.cpu().numpy()
            local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, extend_embs, likelihoods)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose=False)
    result = np.array(decoder.beam[0].words)

    print(f'nuc_max{nuc_max}')
    return result

def gen_word_level_random(length, model,tokenizer, device, Lang, feature_layer = config.llama3_feature_layer):
    nuc_max = 0

    length = length

    llama3 = LLaMA3(model,tokenizer, device)
    # contrxt_words 提取特征的时候上下文长度
    features = LMFeatures(model=llama3, tokenizer=tokenizer, layer=feature_layer, context_words=15)
    # lm = LanguageModel(llama3, nuc_mass=0.9, nuc_ratio=0.03, output_lan=Lang)
    lm = LanguageModel(llama3, nuc_mass=0.9, nuc_ratio=0.03, output_lan=Lang)

    word_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    decoder = Decoder(word_times, 15)
    for sample_index in range(length):
        print(sample_index)
        # ncontext 生成文本的时候用多少上下文预测
        ncontext = 15
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        # beam_nucs = lm.beam_propose_avoidnull(decoder.beam, ncontext)

        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if nuc.__len__() > nuc_max:
                nuc_max = nuc.__len__()

            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))

            likelihoods = np.random.randn(extend_embs.__len__())

            local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, extend_embs, likelihoods)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose=False)
    result = np.array(decoder.beam[0].words)

    print(f'nuc_max{nuc_max}')
    return result

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, prediction, target):
        mse_loss = nn.MSELoss()(prediction, target)
        variance = torch.var(target, unbiased=False)
        nmse_loss = mse_loss / variance
        return nmse_loss

if __name__ == '__main__':
    device = 'cuda:6'
    model_id = "/home/guoyi/llm_model/bloom-1b1"
    # dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        # torch_dtype=dtype,
    )

    start_time = time.time()
    result = gen_word_level_random(60, model, tokenizer, device, 'CN', 20)
    print(f'cost time {time.time() - start_time}')
    print(''.join(list(result)))
