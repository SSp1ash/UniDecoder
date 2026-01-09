from transformers import BertTokenizer, BertModel
import jieba  # 使用jieba进行分词
import scipy.io as scio
import os
import numpy as np
import torch


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

@torch.no_grad()
def BERT_check():
    # 初始化tokenizer和model
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    # 示例文本

    t = '千家万户'
    tt = tokenizer(t, return_tensors='pt', padding=False, add_special_tokens=False)
    ttt = model(**tt, output_hidden_states=True)['hidden_states'][7]
    # eval_test = (torch.sum(ttt,dim=1)/4) .numpy()[0]
    eval_test = torch.max(ttt,dim=1).values.numpy()[0]




    text = "我们经常会说教育关系千家万户，有关教育的讨论总能引发社会关注。"
    # 使用jieba进行分词
    word_tokens = jieba.cut(text)

    # 将分词后的词语重新组合成字符串，以空格分隔
    # tokenized_text = " ".join(word_tokens)
    words = []
    for i in word_tokens:
        words.append(i)
    # 编码分词后的文本
    encoded_input = tokenizer(text, return_tensors='pt',padding=False,add_special_tokens=False)
    output = model(**encoded_input,output_hidden_states=True)

    encoded_input_jieba = tokenizer(words, return_tensors='pt',padding=True,add_special_tokens=False)
    output2 = model(**encoded_input_jieba, output_hidden_states=True)
    # 获取编码后的表示
    embeddings = output['hidden_states'][7]
    embeddings2 = output2['hidden_states'][7]


    eval = torch.sum(embeddings[:, 10:14, :], dim=1)/4
    eval = eval.numpy()[0]
    eval2 = torch.sum(embeddings2[6, :, :], dim=0)/4
    eval2 = eval2.numpy()
    gt = scio.loadmat('../qianjiawanhu.mat')['t'][0,0]
    result = cosine_similarity(eval,gt)
    result2 = cosine_similarity(eval2,gt)
    result3 = cosine_similarity(eval_test,gt)

    print(embeddings)

if __name__ == '__main__':
    # BERT_check()

    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
        path = 'D:\工作文档\博一上\大脑语义解析\数据集\MEG+fMRI听中文短句\ds004078-download\derivatives\\annotations\\time_align\char-level'
        for i in range(1,59):
            data_path = os.path.join(path,f'story_{str(i)}_char_time.mat')
            ts = scio.loadmat(data_path)

            x = "".join(ts['char'][:31])
            x2 = tokenizer(x,return_tensors='pt',add_special_tokens=True)
            em = model(**x2,output_hidden_states=True)
            qianjiawanhu = em['hidden_states'][7][0,11:15,:]
            qianjiawanhu = torch.sum(qianjiawanhu, dim=0) / 4
            qianjiawanhu2 = scio.loadmat('./qianjiawanhu_test_lan_char_sentencesplit.mat')['data']
            qianjiawanhu2 = np.sum(qianjiawanhu2,axis=0)/4
            gt = scio.loadmat('../qianjiawanhu.mat')['t'][0, 0]
            result = cosine_similarity(qianjiawanhu, gt)
            result2 = cosine_similarity(qianjiawanhu2, gt)



            encoded_input = tokenizer(list(ts['char']), return_tensors='pt',padding=True,add_special_tokens=False)
            embed = model(**encoded_input,output_hidden_states=True)['hidden_states'][7]
            time_alingn = (ts['start'] + ts['end']) / 2
            print(123)