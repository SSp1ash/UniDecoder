import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import jieba
import numpy as np


nltk.data.path.append('/home/guoyi/nltk_data')

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

def calculate_bge_score(reference, candidate):
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('/home/guoyi/llm_model/bge-m3',
                           use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

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
    # reference_text = "今天天气很好，我们去公园散步吧"
    # candidate_text = "今天天气不错，我们去公园走走吧"
    # reference_text = "我们经常会说教育关系千家万户，有关教育的讨论总能引发社会关注。最近，不少媒体对云南丽江华坪女子高级中学校长张桂梅的事迹进行了报道，“女校长让1600多名女孩走出大山”感动无数网友。"
    # candidate_text = "我经常会说，教育是每个家庭的难题，对于教育的讨论总是能引发社会关注，近日，不少媒体对云南丽江市华坪县一家小学校长张桂梅的事迹得到报道，而这位校长带领几十名孩子走出大山的故事感动无数网友。"

    # reference_text = "从直播平台购入的洗衣液，到货后发现竟然是小作坊生产的。"
    # candidate_text = "她正在网上购买的护肤品，到货后发现里面全是一些山寨品牌的。"

    # reference_text = "在井旁边有一堵残缺的石墙第二天晚上我工作回来的时候我远远地看见了小王子耷拉着双腿坐在墙上我听见他在说话"
    # candidate_text = "她家门口有一道残缺的石墙。然后我回去的途中，远远的看见了小王子在飘动着他的尾巴在路上他听见他在说话"
    # reference_text = "我们经常会说教育关系千家万户，有关教育的讨论总能引发社会关注。 最近，不少媒体对云南丽江华坪女子高级中学校长张桂梅的事迹进行了报道，“女校长让1600多名女孩走出大山"
    # candidate_text = "他曾经认为，教育是关系着每个人的事，对于教育的讨论，往往成为社会焦点。因此，许多媒体对一些学校里关于学校学生家长和校长们与校园里的事迹的报道，比如一个校长让一个孩子不要"

    reference_text = input('input text A')
    candidate_text = input('input text B')

    bleu = calculate_bleu(reference_text, candidate_text)
    wer = cal_wer(reference_text,candidate_text)
    meteor = calculate_meteor(reference_text,candidate_text)
    bge_score = calculate_bge_score(reference_text,candidate_text)
    print(f"两个句子的WER值为: {wer}")
    print(f"两个句子的BLEU-1值为: {bleu}")
    print(f"两个句子的METEOR值为: {meteor}")
    print(f"两个句子的bge_score值为: {bge_score}")
