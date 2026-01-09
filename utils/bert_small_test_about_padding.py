from transformers import BertTokenizer, BertModel
import jieba  # 使用jieba进行分词
import scipy.io as scio
import os
import numpy as np
import torch
import interpdata
import torch.nn.functional as F

# 得出结论，只要被mask了，后面的值直接去掉就可以
# if __name__ == '__main__':
#     with torch.no_grad():
#         tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#         model = BertModel.from_pretrained('bert-base-chinese')
#
#         text = ['你好','很高兴见到你']
#         input = tokenizer(text,padding=True,return_tensors='pt')
#         emb = model(**input,output_hidden_states=True)
#         print(123)
#
#         mask = input.data['attention_mask']
#         data = emb['hidden_states'][7] * mask[:,:,None]
#         print(123)

# if __name__ == '__main__':
#     text = '北京是中国的首都'
#     with torch.no_grad():
#         tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#         model = BertModel.from_pretrained('bert-base-chinese')
#         input = tokenizer(text, padding=True, return_tensors='pt')
#         emb = model(**input, output_hidden_states=True).hidden_states[7][0]
#         emb = emb[1:-1]
#         torch.mean(emb[0:2],dim=0)
#         emb[2:3]
#         torch.mean(emb[3:5], dim=0)
#         emb[5:6]
#         torch.mean(emb[6:],dim=0)
#         test1 = torch.cat([torch.mean(emb[0:2],dim=0,keepdim=True),emb[2:3],torch.mean(emb[3:5], dim=0,keepdim=True),emb[5:6],torch.mean(emb[6:],dim=0,keepdim=True)],dim=0)
#         print(123)
#         text1 = '北京'
#         text2 = '是'
#         text3 = '中国'
#         text4 = '的'
#         text5 = '首都'
#         input1 = tokenizer(text1, padding=True, return_tensors='pt')
#         input2 = tokenizer(text2, padding=True, return_tensors='pt')
#         input3 = tokenizer(text3, padding=True, return_tensors='pt')
#         input4 = tokenizer(text4, padding=True, return_tensors='pt')
#         input5 = tokenizer(text5, padding=True, return_tensors='pt')
#         emb1 = model(**input1, output_hidden_states=True).hidden_states[7][0][1:-1].mean(0).unsqueeze(0)
#         emb2 = model(**input2, output_hidden_states=True).hidden_states[7][0][1:-1].mean(0).unsqueeze(0)
#         emb3 = model(**input3, output_hidden_states=True).hidden_states[7][0][1:-1].mean(0).unsqueeze(0)
#         emb4 = model(**input4, output_hidden_states=True).hidden_states[7][0][1:-1].mean(0).unsqueeze(0)
#         emb5 = model(**input5, output_hidden_states=True).hidden_states[7][0][1:-1].mean(0).unsqueeze(0)
#         test2 = torch.cat([emb1,emb2,emb3,emb4,emb5],dim=0)
#
#         print(F.cosine_similarity(test1, test2))
if __name__ == '__main__':
    text = '北京是中国的首都'
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
        input = tokenizer(text, padding=True, return_tensors='pt',add_special_tokens=True)
        emb = model(**input, output_hidden_states=True).hidden_states[7][0]
        test1 = emb[1:-1]
        # test2 = []
        # for i in text:
        #     input = tokenizer(i, padding=True, return_tensors='pt', add_special_tokens=True)
        #     emb = model(**input, output_hidden_states=True).hidden_states[7][0][1]
        #     test2.append(emb)
        # test2 = torch.stack(test2)
        text2 = '北京是中国的首都'
        input2 = tokenizer(text2, padding=True, return_tensors='pt', add_special_tokens=False)
        emb = model(**input2, output_hidden_states=True).hidden_states[7][0]
        test2 = emb


        print(F.cosine_similarity(test1, test2))




