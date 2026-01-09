import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import jieba
import numpy as np
import config
import pandas as pd
import config
import glob
import re
nltk.data.path.append('nltk_path')

def segment_data(data, times, cutoffs):
    return [[x for c, x in zip(times, data) if c >= start and c < end] for start, end in cutoffs]

def windows(start_time, end_time, duration, step = 1):
    start_time, end_time = int(start_time), int(end_time)
    half = int(duration / 2)
    return [(center - half, center + half) for center in range(start_time + half, end_time - half + 1) if center % step == 0]
def cal_wer(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER) between reference and hypothesis.
    Parameters:
        reference (str): Reference sentence.
        hypothesis (str): Hypothesis sentence.
    Returns:
        float: WER value.
    """
    # Split sentences into words for English
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Create the distance matrix
    r_len = len(ref_words)
    h_len = len(hyp_words)
    d = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)

    # Initialize distance matrix
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    # Populate the distance matrix
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,      # Deletion
                          d[i][j - 1] + 1,      # Insertion
                          d[i - 1][j - 1] + cost)  # Substitution

    # Compute WER
    wer_value = d[r_len][h_len] / r_len
    return wer_value

def calculate_bleu(reference, candidate):
    # Split sentences into words for English
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    # BLEU calculation requires reference translations as a nested list
    reference_list = [reference_tokens]

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_list, candidate_tokens, weights=(1.0, 0, 0, 0))
    return bleu_score

def calculate_meteor(reference, candidate):
    # Split sentences into words for English
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    # Calculate METEOR score
    meteor_score = single_meteor_score(reference_tokens, candidate_tokens)
    return meteor_score

from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
def calculate_bge_score(reference, candidate):

    sentences_1 = reference
    sentences_2 = candidate

    embeddings_1 = model.encode(sentences_1,
                                batch_size=12,
                                max_length=8192,
                                # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    embeddings_2 = model.encode(sentences_2)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    return similarity

def windows_compare():
    columns = ['Subject_ID', 'Story_no','WER', 'BLEU', 'METEOR', 'BGE_Score']
    results_df = pd.DataFrame(columns=columns)
    story_list = [9]

    for story in story_list:

        story_list = glob.glob(config.result_path + f'/largebatch/LPPC-fMRI/*.npy')
        sub_list = sorted(
            [item for item in story_list if re.search(r'subEN\d+', item)],  # 筛选出 sub_EN 开头的路径
            key=lambda x: int(re.search(r'subEN(\d+)', x).group(1))  # 提取 EN 后面的数字用于排序
        )
        # sub_list = [re.search(r'subEN(\d+)', path).group(1) for path in sub_list if re.search(r'subEN(\d+)', path)]

        for sub_path in sub_list:
            bleu_list = []
            wer_list = []
            meteor_list = []
            bge_score_list = []

            # rec = np.load(
            #     config.result_path + f'/largebatch/LPPC-fMRI/LPPC-fMRI_{i}_test_overfit_20241212_text.npy')
            rec = np.load(sub_path)
            sub_no = re.search(r'sub(EN\d+)', sub_path).group(1)
            gt = np.load('./word_compare_gt/story9_EN.npy')[:-4]
            time = np.load('./word_times_bloom/word_times_storyrun{story}.npy')[:-4]



            window_cutoffs = windows(time[0],time[-1], 20, step=5)


            rec_windows = segment_data(rec,time,window_cutoffs)
            gt_windows = segment_data(gt,time,window_cutoffs)

            rec_windows_str = [' '.join(rec_windows[i]) for i in range(rec_windows.__len__())]
            gt_windows_str = [' '.join(gt_windows[i]) for i in range(gt_windows.__len__())]


            for win_no in range(rec_windows_str.__len__()):
                reference_text = rec_windows_str[win_no]
                candidate_text = gt_windows_str[win_no]

                bleu = calculate_bleu(reference_text, candidate_text)
                wer = cal_wer(reference_text, candidate_text)
                meteor = calculate_meteor(reference_text, candidate_text)
                bge_score = calculate_bge_score(reference_text, candidate_text)

                print(f"WER: {wer}")
                print(f"BLEU-1: {bleu}")
                print(f"METEOR: {meteor}")
                print(f"bge_score: {bge_score}")


if __name__ == "__main__":

    windows_compare()
