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
nltk.data.path.append('/home/guoyi/nltk_data')

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
    # Split sentences into words using jieba for Chinese
    ref_words = list(jieba.cut(reference))
    hyp_words = list(jieba.cut(hypothesis))

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
    # 使用jieba对中文字符串进行分词
    reference_tokens = list(jieba.cut(reference))
    candidate_tokens = list(jieba.cut(candidate))

    # BLEU计算需要将参考翻译作为嵌套列表传入
    reference_list = [reference_tokens]

    # 计算BLEU值
    bleu_score = sentence_bleu(reference_list, candidate_tokens, weights = (1.0, 0, 0, 0))
    return bleu_score

def calculate_meteor(reference, candidate):
    # 使用jieba对中文字符串进行分词
    reference_tokens = list(jieba.cut(reference))
    candidate_tokens = list(jieba.cut(candidate))

    # 计算METEOR值
    meteor_score = single_meteor_score(reference_tokens, candidate_tokens)
    return meteor_score

from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('/home/guoyi/llm_model/bge-m3',
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

def global_compare():
    columns = ['Subject_ID', 'Story_no','WER', 'BLEU', 'METEOR', 'BGE_Score']
    results_df = pd.DataFrame(columns=columns)
    story_list = [60]

    for story in story_list:
        # 全局对比
        # 随机值生成?
        # LPPC-fMRI 12 受试者
        start = 1
        end = 12
        for i in range(start, end + 1):
            rec = np.load(
                config.result_path + f'/largebatch/LPPC-fMRI_fMRI/story{story}/LPPC-fMRI_sub{str(i).zfill(2)}_test_20241212_text.npy')
            gt = np.load(config.project_lfs_path + f'/LPPC-fMRI/dataset/word_compare_gt/story{story}.npy')
            rec_str = ''.join(list(rec))
            gt_str = ''.join(list(gt))

            # reference_text = input('input text A')
            # candidate_text = input('input text B')
            reference_text = gt_str
            candidate_text = rec_str

            bleu = calculate_bleu(reference_text, candidate_text)
            wer = cal_wer(reference_text, candidate_text)
            meteor = calculate_meteor(reference_text, candidate_text)
            bge_score = calculate_bge_score(reference_text, candidate_text)
            print(f"两个句子的WER值为: {wer}")
            print(f"两个句子的BLEU-1值为: {bleu}")
            print(f"两个句子的METEOR值为: {meteor}")
            print(f"两个句子的bge_score值为: {bge_score}")

            result_row = pd.DataFrame([{
                'Subject_ID': f'Sub{i}',
                'Story_no': f'story{story}',
                'WER': wer,
                'BLEU': bleu,
                'METEOR': meteor,
                'BGE_Score': bge_score
            }])

            # 使用pd.concat()将新的一行结果添加到原始DataFrame
            results_df = pd.concat([results_df, result_row], ignore_index=True)
    results_df.to_csv(config.result_path + '/largebatch/excels/LPPC-fMRI_fMRI_singlesubmodel.csv', index=False,
                          encoding='utf-8')


def windows_compare():
    columns = ['Subject_ID', 'Story_no','WER', 'BLEU', 'METEOR', 'BGE_Score']
    results_df = pd.DataFrame(columns=columns)
    story_list = [9]

    for story in story_list:
        # 全局对比，时间窗口对比
        # 随机值生成
        # LPPC-fMRI CN受试者
        story_list = glob.glob(config.result_path + f'/largebatch/LPPC-fMRI/*.npy')
        sub_list = sorted(
            [item for item in story_list if re.search(r'subCN\d+', item)],  # 筛选出 sub_CN 开头的路径
            key=lambda x: int(re.search(r'subCN(\d+)', x).group(1))  # 提取 CN 后面的数字用于排序
        )
        # sub_list = [re.search(r'subCN(\d+)', path).group(1) for path in sub_list if re.search(r'subCN(\d+)', path)]

        for sub_path in sub_list:
            bleu_list = []
            wer_list = []
            meteor_list = []
            bge_score_list = []

            # rec = np.load(
            #     config.result_path + f'/largebatch/LPPC-fMRI/LPPC-fMRI_{i}_test_20241212_text.npy')
            rec = np.load(sub_path)
            sub_no = re.search(r'sub(CN\d+)', sub_path).group(1)
            gt = np.load(config.project_lfs_path + f'/LPPC-fMRI/dataset/word_compare_gt/story{story}_CN.npy')[:-4]
            time = np.load(config.project_lfs_path + f'/LPPC-fMRI/dataset/word_times_bloom/word_times_storyrun{story}.npy')[:-4]


            # 根据stroy选择时间
            window_cutoffs = windows(time[0],time[-1], 20, step=5)


            rec_windows = segment_data(rec,time,window_cutoffs)
            gt_windows = segment_data(gt,time,window_cutoffs)

            rec_windows_str = [''.join(rec_windows[i]) for i in range(rec_windows.__len__())]
            gt_windows_str = [''.join(gt_windows[i]) for i in range(gt_windows.__len__())]


            for win_no in range(rec_windows_str.__len__()):
                reference_text = rec_windows_str[win_no]
                candidate_text = gt_windows_str[win_no]

                bleu = calculate_bleu(reference_text, candidate_text)
                wer = cal_wer(reference_text, candidate_text)
                meteor = calculate_meteor(reference_text, candidate_text)
                bge_score = calculate_bge_score(reference_text, candidate_text)

                bleu_list.append(bleu)
                wer_list.append(wer)
                meteor_list.append(meteor)
                bge_score_list.append(bge_score)


            result_row = pd.DataFrame([{
                'Subject_ID': f'Sub{sub_no}',
                'Story_no': f'story{story}',
                'WER': np.mean(wer_list),
                'BLEU': np.mean(bleu_list),
                'METEOR': np.mean(meteor_list),
                'BGE_Score': np.mean(bge_score_list)
            }])


            # 使用pd.concat()将新的一行结果添加到原始DataFrame
            results_df = pd.concat([results_df, result_row], ignore_index=True)
            print(f'sub_{sub_no} finish')
    results_df.to_csv(config.result_path + '/largebatch/excels/LPPC-fMRI/fMRI_singlesubmodel_windows_CN.csv', index=False,
                      encoding='utf-8')

if __name__ == "__main__":
    # global_compare()
    windows_compare()
