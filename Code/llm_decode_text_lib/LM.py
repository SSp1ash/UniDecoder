import time
import torch
import numpy as np
import json
import re
import langdetect

import config

DEFAULT_BAD_WORDS = frozenset(["，", "。", "”", "“", "、", "：", "？",'[',']'])


def filter_chinese_only(input_list):
    # 定义中文字符的正则表达式
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    # 过滤只包含中文字符的内容
    result = [item for item in input_list if chinese_pattern.search(item)]
    return result

def filter_chinese_only_opt(input_list):
    # 定义中文字符的正则表达式
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    result = [item for item in input_list if chinese_pattern.search(item)]
    if result.__len__() == 0:
        result.append('。')
    # 过滤只包含中文字符的内容
    return result

def filter_chinese(text_list):
    # 使用正则表达式匹配中文字符、中文逗号（，）和中文句号（。）
    pattern = re.compile(r'^[\u4e00-\u9fa5，。]+$')
    # 过滤掉不符合要求的文本
    filtered_list = [text for text in text_list if pattern.match(text)]
    return filtered_list

# def filter_french(text_list):
#     # 匹配法语字符（包含重音字符）、空格和常用标点符号
#     pattern = re.compile(r'^[a-zA-ZÀ-ÿ0-9\s.,!?\'\"-]+$')
#     # 过滤掉不符合要求的文本
#     filtered_list = [text for text in text_list if pattern.match(text)]
#     return filtered_list


def filter_french(text_list):
    # 匹配法语字符（包括带重音的字符）、空格和常用标点符号
    pattern = re.compile(r'^[a-zA-ZÀ-ÿ\s.,!?\'\"-]+$')
    # 过滤掉不符合要求的文本
    filtered_list = [text for text in text_list if pattern.match(text)]
    return filtered_list


def filter_NL(text_list):
    filtered_list = []
    # 加载荷兰语词典
    def load_dutch_dictionary(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip())
    # 指定荷兰语词典的绝对路径
    dictionary_path = config.project_lfs_path + '/dictionary/OpenTaal-210G-basis-gekeurd.txt'  # 替换为实际的词典文件路径

    # 加载词典
    dutch_dict = load_dutch_dictionary(dictionary_path)

    # 过滤荷兰语单词
    filtered_list = [word for word in text_list if word.strip().lower() in dutch_dict]

    return filtered_list


def get_nucleus(probs, nuc_mass, nuc_ratio):
    """identify words that constitute a given fraction of the probability mass
    """
    nuc_ids = np.where(probs >= np.max(probs) * nuc_ratio)[0]
    nuc_pairs = sorted(zip(nuc_ids, probs[nuc_ids]), key = lambda x : -x[1])
    nuc_pairs = nuc_pairs[:200]

    sum_mass = np.cumsum([x[1] for x in nuc_pairs])
    cutoffs = np.where(sum_mass >= nuc_mass)[0]
    if len(cutoffs) > 0: nuc_pairs = nuc_pairs[:cutoffs[0]+1]
    nuc_ids = [x[0] for x in nuc_pairs]
    return nuc_ids

def check_legal(x):
    if (x > 670 and x < 7992) or x in [511,8024]:
    # if x > 670 and x < 7992:
        return True
    else:
        return False



# def in_context(word, context):
#     """test whether [word] or a stem of [word] is in [context]
#     """
#     stem_context = [stemmer.stem(x) for x in context]
#     stem_word = stemmer.stem(word)
#     return (stem_word in stem_context or stem_word in context)

def context_filter(proposals, context):
    """filter out words that occur in a context to prevent repetitions
    """
    cut_words = ['']

    # 防止重复
    cut_words.extend(context[-1])

    # 防止进入死循环，比如，我把我把我把我把，遍历除了最后一个字符的整个句子，如果遇到与最后一个字符相等，就把下一个加入屏蔽词
    cut_words.extend([context[i+1] for i, word in enumerate(context[:-1]) if word == context[-1]]) # bigrams
    # 防止重复的名词
    # cut_words.extend([x for x in proposals if x not in STOPWORDS and in_context(x, context)]) # unigrams
    return [x for x in proposals if x not in cut_words]

class LanguageModel():
    """class for generating word sequences using a language model
    """

    def __init__(self, model, nuc_mass=1.0, nuc_ratio=0.0,output_lan='CN'):
        self.model = model
        # self.ids = {i for word, i in self.model.word2id.items() if word in set(vocab)}
        self.nuc_mass, self.nuc_ratio = nuc_mass, nuc_ratio
        self.output_lan = output_lan
        # with open("./vocab.json", "r", encoding='utf-8') as f:
        #     self.gpt_vocab = json.load(f)
        if output_lan == 'CN':
            self.INIT = ['我','你','他','她']
            # self.INIT = ['我','你','他','她','它','这','在','当']
            # self.INIT = ['当','在','这']
            # self.prompt = self.model.tokenizer.encode('请用中文继续生成冒号后的文本：', return_tensors='pt')
            self.prompt = self.model.tokenizer.encode('请在冒号后面继续生成文本，只能生成中文：', return_tensors='pt')

        if output_lan == 'EN':
            self.INIT = ['I', 'You', 'He', 'She', 'It', 'This','The']
            # self.prompt = self.model.tokenizer.encode('请用中文继续生成冒号后的文本：', return_tensors='pt')
            # self.prompt = self.model.tokenizer.encode('Please continue to generate the text after the colon in English:', return_tensors='pt')
            self.prompt = self.model.tokenizer.encode('Please continue generating text after the colon, only English can be generated:', return_tensors='pt')

            # self.prompt = self.model.tokenizer.encode('Generate text based on what follows the colon, with content in English:', return_tensors='pt')

        if output_lan== 'FR':
            self.INIT = ['Je', 'Tu', 'Il', 'Elle', 'cela']
            # self.INIT = ['Je', 'tu', 'il', 'elle', 'ça']
            self.prompt = self.model.tokenizer.encode('Générer un texte basé sur ce qui suit les deux points, avec un contenu français:', return_tensors='pt')

        if output_lan == 'NL':
            # self.INIT = ['Ik', 'Jij', 'Hij', 'Zij', 'Dit', 'Dat', 'De', 'Een']
            # self.INIT = ['Ik', 'Dit', 'Dat', 'De','Je', 'Ze', 'het']
            self.INIT = ['Ik', 'Dat', 'De', 'Je', 'Ze', 'het', 'En']
            self.prompt = self.model.tokenizer.encode(
                'Ga verder met het genereren van tekst na de dubbele punt, alleen in het Nederlands:',
                return_tensors='pt')

    def ps(self, contexts):
        """get probability distributions over the next words for each context
        """
        context_arr = self.model.get_context_array(contexts)
        # 添加中文prompt
        # 用中文续写文本：  [11883, 108891, 105196, 62543, 17161, 22656, 5232]
        # prompt = torch.tensor([11883, 108891, 105196, 62543, 17161, 22656, 5232])
        # self.prompt = torch.tensor([15225,  11883, 108891, 114638,  45059, 112798,  18476, 121964,  17161, 22656, 5232])
        context_arr = torch.cat((self.prompt.repeat(context_arr.size(0), 1),context_arr), dim=1)
        probs = self.model.get_probs(context_arr)
        return probs[:, -1]

    def beam_propose(self, beam, context_words):
        """get possible extension words for each hypothesis in the decoder beam
        """
        t1 = time.time()
        if len(beam) < 2:
            # nuc_words = [w for w in INIT if self.model.word2id[w] in self.ids]
            nuc_words = self.INIT
            nuc_logprobs = np.log(np.ones(len(nuc_words)) / len(nuc_words))
            return [(nuc_words, nuc_logprobs)]
        else:
            contexts = [hyp.words[-context_words:] for hyp in beam]

            beam_probs = self.ps(contexts)
            beam_nucs = []
            for context, probs in zip(contexts, beam_probs):
                nuc_ids = get_nucleus(probs, nuc_mass=self.nuc_mass, nuc_ratio=self.nuc_ratio)
                # 先不加标点符号，去掉了标点符号
                # nuc_words = [word for word in (self.model.tokenizer.decode(int(i)) for i in nuc_ids) if word in self.gpt_vocab]
                # if self.output_lan == 'CN':
                #     nuc_ids = [i for i in nuc_ids if check_legal_chinese(i)]

                # nuc_words = self.model.tokenizer.decode(nuc_ids, clean_up_tokenization_spaces=False)
                # nuc_words = nuc_words.split(' ')

                nuc_words = [self.model.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in nuc_ids]

                # 过滤非中文
                # nuc_words = filter_chinese_only_opt(nuc_words)
                # nuc_words = filter_chinese_only(nuc_words)
                if self.output_lan == 'CN':
                    nuc_words = filter_chinese(nuc_words)
                if self.output_lan =='FR':
                    nuc_words = filter_french(nuc_words)

                if self.output_lan == 'NL':
                    nuc_words = filter_NL(nuc_words)

                if nuc_words == [""]:
                    nuc_words = []
                nuc_words = context_filter(nuc_words, context)
                nuc_logprobs = np.log([probs[self.model.encode(w)[0]] for w in nuc_words])
                beam_nucs.append((nuc_words, nuc_logprobs))
            return beam_nucs

