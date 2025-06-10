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
import sys
import os


if __name__ == '__main__':
    run_eval = 1

    with torch.no_grad():
        # 设置映射到大模型哪一层
        layer = 20
        roi = extract_speech(config.project_lfs_path + r'/LPPC-fMRI/CN001.aparc.a2009s.32k_fs_LR.dlabel.nii')
        vox = roi.__len__()

        # lan_list = ['CN','EN','FR']
        device = 'cuda:3'
        model_id = "/home/guoyi/llm_model/bloom-1b1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        split = 15

        # lan_list = ['CN']
        lan_list = ['FR']


        for lan in lan_list:
            data_info_all = pd.read_csv(config.rawdata_path + fr'/LPPC-fMRI/ds003643-download/annotation/{lan}/lpp{lan}_word_information.csv')
            emb_train_list = []
            fmri_resample_train_list = []
            fmri_train_list = []
            wr_train_list = []
            eb_downsample_train_list = []
            with open(config.rawdata_path + fr'/LPPC-fMRI/ds003643-download/annotation/{lan}/TR.json', 'r', encoding='utf-8') as json_file:
                TR = json.load(json_file)

            i = 0

            TRs = np.array(list(TR.values()))
            for run, TR_num in TR.items():
                run_no = int(run[-1])
                index = (data_info_all['section'] == run_no)
                data_info = data_info_all[index]
                time_alingn = np.array((data_info['onset'] + data_info['offset']) / 2)

                # 改为字符级别的编码
                words = np.array(data_info['word'])
                chars, char_time = split_words_and_times(words, time_alingn)
                char_time = char_time[np.newaxis,:]
                text = ''.join(list(chars))
                sys.stdout = open(os.devnull, 'w')
                token_ids, tokens, offset_mapping = llama3_fenci(tokenizer,text)
                sys.stdout = sys.__stdout__
                if run_no == run_eval:
                    result = ''.join(tokens[:37])
                    print(result)

