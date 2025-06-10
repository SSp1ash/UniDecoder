from transformers import BertTokenizer, BertModel
import jieba  # 使用jieba进行分词
import scipy.io as scio
import os
import numpy as np
import torch
import interpdata
import re
import feature_convolve
from nilearn.glm.first_level import hemodynamic_models


# 时间对齐方案采用lanczos
# 测试char_embedding的
@torch.no_grad()
def test_lan_char():

    # 实验结果相关性为0.58，模型是用word_embedding训练，而word_embedding数据为0.74.
    # 嵌入的千家万户词语，经过与原始word_embedding比较，相关性为0.82
    # 全部字符输入给BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\char-level'
    data_path = os.path.join(path, f'story_{str(1)}_char_time.mat')
    ts = scio.loadmat(data_path)
    time_alingn = (ts['start'] + ts['end']) / 2

    text = ''.join(ts['char'])
    input = tokenizer.encode(text, return_tensors='pt')
    # input = tokenizer(text,return_tensors='pt',padding=False,add_special_tokens=True,max_length=512)
    input1 = input[...,0:512]
    input2 = input[...,512:1024]
    input3 = input[...,1024:1536]
    input4 = input[...,1536:]
    emb1 = model(input1,output_hidden_states=True)['hidden_states'][7]
    emb2 = model(input2,output_hidden_states=True)['hidden_states'][7]
    emb3 = model(input3,output_hidden_states=True)['hidden_states'][7]
    emb4 = model(input4,output_hidden_states=True)['hidden_states'][7]
    emb_fin = torch.concatenate([emb1,emb2,emb3,emb4],dim=1)[0]
    emb_fin = emb_fin.numpy()[1:-1]
    bold_time = np.arange(0, int(614) * 0.71, 0.71)[:int(614)]

    temp = interpdata.lanczosinterp2D(emb_fin, time_alingn[0], bold_time)
    np.savez_compressed('./char_embedding_lanczos',temp)

    # scio.savemat('qianjiawanhu_test_lan_char.mat',{'data':emb1[0,11:15].numpy()})

    # #按照句子输入给BERT
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # model = BertModel.from_pretrained('bert-base-chinese')
    # path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\char-level'
    # data_path = os.path.join(path, f'story_{str(1)}_char_time.mat')
    # ts = scio.loadmat(data_path)
    # time_alingn = (ts['start'] + ts['end']) / 2
    #
    # text = ''.join(ts['char'])
    # text = text.replace(' ', '')
    #
    # sentences = re.split(r'(?<=[。！？])', text)
    # sentences = [s.strip() for s in sentences if s.strip() != '']
    #
    # # word_tokens = jieba.cut(text)
    # # jieba_text = " ".join(word_tokens)
    #
    # input = tokenizer(sentences, return_tensors='pt', padding=True)
    # # input = tokenizer(text,return_tensors='pt',padding=False,add_special_tokens=True,max_length=512)
    # emb = model(**input, output_hidden_states=True)['hidden_states'][7]
    #
    # mask = input.data['attention_mask'][:, :, None]
    # mask = mask.bool()
    # mask[:, 0] = 0  # 去除CLS
    # sep_positions = (input.data['input_ids'] == 102)
    # mask[sep_positions] = 0  # 去除SEP
    #
    # selected_embeddings = torch.masked_select(emb, mask)
    # selected_embeddings = selected_embeddings.view(-1, 768)
    #
    # bold_time = np.arange(0, int(614) * 0.71, 0.71)[:int(614)]
    #
    # temp = interpdata.lanczosinterp2D(selected_embeddings, time_alingn[0], bold_time)
    # np.savez_compressed('./char_embedding_lanczos_sentencesplit', temp)

# 测试word_embedding
@torch.no_grad()
def test_lan_word():
    #因为全部给BERT效果不好，或者说是和训练的模型有背，后面模型实验都是以句子为单位输入
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\char-level'
    data_path = os.path.join(path, f'story_{str(1)}_char_time.mat')
    ts = scio.loadmat(data_path)
    time_alingn = (ts['start'] + ts['end']) / 2
    time_alingn = time_alingn[0]


    text = ''.join(ts['char'])
    text = text.replace(' ','')

    sentences = re.split(r'(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if s.strip() != '']

    input = tokenizer(sentences, return_tensors='pt',padding=True)
    # input = tokenizer(text,return_tensors='pt',padding=False,add_special_tokens=True,max_length=512)
    emb = model(**input, output_hidden_states=True)['hidden_states'][7]

    mask = input.data['attention_mask'][:,:,None]
    mask = mask.bool()
    mask[:,0] = 0  # 去除CLS
    sep_positions = (input.data['input_ids']==102)
    mask[sep_positions] = 0 #去除SEP

    # 得到以句子级别输入BERT的char-level-embedding，接下来需要用jieba分词
    selected_embeddings = torch.masked_select(emb, mask)
    selected_embeddings = selected_embeddings.view(-1,768)


    bold_time = np.arange(0, int(614) * 0.71, 0.71)[:int(614)]




    # 临时解决办法，在故事1中，将长度大于1的数字进行裁剪
    text = text.replace('1600', '1')
    text = text.replace('63', '1')
    text = text.replace('18', '1')
    text = text.replace('11', '1')
    word_embeddings = []
    word_time = []
    tokens_with_indices = list(jieba.tokenize(text))
    for token, start, end in tokens_with_indices:
        # 提取当前词的嵌入（start到end-1）
        token_embeddings = selected_embeddings[start:end]

        word_time.append(time_alingn[start:end].mean())
        # 计算当前词的嵌入的平均值
        token_embedding_mean = token_embeddings.mean(dim=0)
        # 保存结果
        word_embeddings.append(token_embedding_mean)

    # 将结果转换为张量
    word_embeddings_tensor = torch.stack(word_embeddings).numpy()
    word_time = np.array(word_time)

    temp = interpdata.lanczosinterp2D(word_embeddings_tensor, word_time, bold_time)
    np.savez_compressed('./word_embedding_lanczos_sentencesplit', temp)




# 时间对齐方案采用HRF
# 测试char_embedding
@torch.no_grad()
def test_hrf_char():
    # 按照句子输入给BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\char-level'
    data_path = os.path.join(path, f'story_{str(1)}_char_time.mat')
    ts = scio.loadmat(data_path)
    time_alingn = (ts['start'] + ts['end']) / 2

    text = ''.join(ts['char'])
    text = text.replace(' ', '')

    sentences = re.split(r'(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if s.strip() != '']

    # word_tokens = jieba.cut(text)
    # jieba_text = " ".join(word_tokens)

    input = tokenizer(sentences, return_tensors='pt', padding=True)
    # input = tokenizer(text,return_tensors='pt',padding=False,add_special_tokens=True,max_length=512)
    emb = model(**input, output_hidden_states=True)['hidden_states'][7]

    mask = input.data['attention_mask'][:, :, None]
    mask = mask.bool()
    mask[:, 0] = 0  # 去除CLS
    sep_positions = (input.data['input_ids'] == 102)
    mask[sep_positions] = 0  # 去除SEP

    selected_embeddings = torch.masked_select(emb, mask)
    selected_embeddings = selected_embeddings.view(-1, 768)

    bold_time = np.arange(0, int(614) * 0.71, 0.71)[:int(614)]


    # temp = interpdata.lanczosinterp2D(selected_embeddings, time_alingn[0], bold_time)
    hrf = hemodynamic_models.spm_hrf(0.71,oversampling=1)
    temp = feature_convolve.convolve_downsample(selected_embeddings,time_alingn,hrf,0,614)


    np.savez_compressed('./char_embedding_HRF_sentencesplit', temp)

    print(123)


# 测试word_embedding
@torch.no_grad()
def test_hrf_word():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    print(123)