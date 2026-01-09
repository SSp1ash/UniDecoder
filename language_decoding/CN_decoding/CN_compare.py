import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import jieba
import numpy as np
import pandas as pd

nltk.data.path.append('nltk_path')


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
    # using jieba chinese fenci
    reference_tokens = list(jieba.cut(reference))
    candidate_tokens = list(jieba.cut(candidate))


    reference_list = [reference_tokens]


    bleu_score = sentence_bleu(reference_list, candidate_tokens, weights = (1.0, 0, 0, 0))
    return bleu_score

def calculate_meteor(reference, candidate):

    reference_tokens = list(jieba.cut(reference))
    candidate_tokens = list(jieba.cut(candidate))


    meteor_score = single_meteor_score(reference_tokens, candidate_tokens)
    return meteor_score

def calculate_bge_score(reference, candidate):
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('/home/guoyi/llm_model/bge-m3',
                           use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # model = BGEM3FlagModel('BAAI/bge-m3',
    #                        use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

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

if __name__ == "__main__":
    columns = ['Subject_ID', 'WER', 'BLEU', 'METEOR', 'BGE_Score']
    results_df = pd.DataFrame(columns=columns)


    rec = np.load('./Decoded_story_text.npy')
    gt = np.load('./word_compare_gt/story60.npy')
    rec_str = ''.join(list(rec))
    gt_str = ''.join(list(gt))


    reference_text = gt_str
    candidate_text = rec_str


    bleu = calculate_bleu(reference_text, candidate_text)
    wer = cal_wer(reference_text,candidate_text)
    meteor = calculate_meteor(reference_text,candidate_text)
    bge_score = calculate_bge_score(reference_text,candidate_text)
    print(f"WER: {wer}")
    print(f"BLEU-1: {bleu}")
    print(f"METEOR: {meteor}")
    print(f"bge_score: {bge_score}")

