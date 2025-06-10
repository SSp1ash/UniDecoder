import numpy as np
BASE_PATH = 'D:/工作文档/博一上/大脑语义解析/数据集/MEG+fMRI听中文短句/ds004078-download/derivatives/'
# BASE_PATH = '/media/test/data/gy/chinese_speech'

RESULT_DIR = './result'
DATA_TRAIN_DIR = BASE_PATH + 'preprocessed_data'
DATA_TEST_DIR = BASE_PATH + 'preprocessed_data'
WORDSEQS_DIR = BASE_PATH + 'annotations/time_align/char-level-process'
# WORDSEQS_DIR = BASE_PATH + 'annotations/time_align/word-level'


WORD2VEC_EXIST_DIR = BASE_PATH + 'annotations/embeddings/word2vec/char-level/300d'
MODEL_DIR_SAVE = './model/'
REPO_DIR = './repo'

# Encoding model parameters
GPT_PATH = 'C:/Users/guoyi/albert-base-chinese-cluecorpussmall/'
# GPT_PATH = './albert-base-chinese-cluecorpussmall/'
# GPT_PATH = 'uer/gpt2-chinese-cluecorpussmall'
VOCAB_PATH = GPT_PATH + 'vocab.txt'

# GPT encoding model paramters
TR = 0.71
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 50
CHUNKLEN = 40
VOXELS = 10000
GPT_LAYER = 9
GPT_WORDS = 5

# decoder parateters
RANKED = True
WIDTH = 50
NM_ALPHA = 2/3
LM_TIME = 10
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# evaluation parameters

WINDOW = 20

# device
GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"