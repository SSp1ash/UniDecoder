from base_decode_text_lib.utils_eval import generate_null, load_transcript, windows, segment_data, WER, BLEU, METEOR, BERTSCORE, SENTENCE_SCORE
import argparse
import numpy as np
import scipy.io as scio
import re
import pandas as pd
import json

def contains_punctuation(s):
    return bool(re.search(r'[^\w\s]', s))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    char_path = '/media/test/data/gy/chinese_speech/annotations/time_align/char-level/'

    # parser.add_argument("--metrics", nargs="+", type=str, default=["BLEU", "METEOR", "BERT"])
    # parser.add_argument("--metrics", nargs="+", type=str, default=["SENTENCE_SCORE"])
    parser.add_argument("--metrics", nargs="+", type=str, default=["BERT"])
    args = parser.parse_args()

    metrics = {}
    if "WER" in args.metrics: metrics["WER"] = WER(use_score = True)
    if "BLEU" in args.metrics: metrics["BLEU"] = BLEU(n = 1)
    if "METEOR" in args.metrics: metrics["METEOR"] = METEOR()
    if "BERT" in args.metrics: metrics["BERT"] = BERTSCORE(
        rescale = False,
        score = "recall")
    if "SENTENCE_SCORE" in args.metrics: metrics["SENTENCE_SCORE"] = SENTENCE_SCORE()

    # 读取gt SMN4Lang
    # data = scio.loadmat(char_path + f'story_1_char_time.mat')
    # data = scio.loadmat(char_path + f'story_58_char_time.mat')
    data = scio.loadmat(char_path + f'story_60_char_time.mat')
    char = data['char']
    char = [s[0] for s in list(char)]
    # char = ''.join(char)
    gt = np.array(char)[:100]


    # 读取gt LPPC-fMRI
    lan = 'CN'
    run_no = 9
    data_info_all = pd.read_csv(
        fr'/media/test/data/gy/ds003643-download/annotation/{lan}/lpp{lan}_word_information.csv')
    with open(fr'/media/test/data/gy/ds003643-download/annotation/{lan}/TR.json', 'r', encoding='utf-8') as json_file:
        TR = json.load(json_file)
    index = (data_info_all['section'] == run_no)
    data_info = data_info_all[index]
    words = np.array(data_info['word'])
    gt = ''.join(list(words))
    gt = gt[:100]



    # SMN4Lang
    # pred_meg = np.load('./story60_fragment_meg.npy')
    # pred_fmri = np.load('./story60_fragment_fMRI.npy')
    # pred_multi = np.load('./story60_fragment_multi.npy')
    # pred_gt = np.load('./story60_fragment_gt.npy')[:100]
    # null = gen_null(100)

    # pred_fmri_pipeline = np.load('./SMN4Lang_char_level_fMRI_upsample/story60_fragment_fMRI_pipeline.npy')

    # LPPC-fMRI
    pred_fmri = np.load('./LPPC-fMRI_story9_fragment_fMRI.npy')
    pred_gt = np.load('./LPPC-fMRI_story9_fragment_gt.npy')

    pred_fmri_pipeline = np.load('./LPPC-fMRI_char_level_fMRI_upsample/LPPC-fMRI_subCN001_story9_fMRI_pipeline.npy')[:100]
    # null = gen_null(100)

    # gt = np.array([item for item in gt if not contains_punctuation(item)])
    # gt_eb = np.array([item for item in gt_eb if not contains_punctuation(item)])
    # pred = np.array([item for item in pred if not contains_punctuation(item)])
    # null = np.array([item for item in null if not contains_punctuation(item)])

    for mname, metric in metrics.items():
        # result_null = metric.score(ref=gt, pred=null)
        result_gteb = metric.score(ref=gt, pred=pred_gt)
        result_fmri = metric.score(ref=gt, pred=pred_fmri)
        # result_meg = metric.score(ref=gt, pred=pred_meg)
        # result_multi = metric.score(ref=gt, pred=pred_multi)

        result_fmri_pipeline = metric.score(ref=gt, pred=pred_fmri_pipeline)
        print(123)