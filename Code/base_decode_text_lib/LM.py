import time

import numpy as np
import json

# INIT = ['i', 'we', 'she', 'he', 'they', 'it']
INIT = ['我','你','他','她','它','这','在']
DEFAULT_BAD_WORDS = frozenset(["，", "。", "”", "“", "、", "：", "？",'[',']'])



def get_nucleus(probs, nuc_mass, nuc_ratio):
    """identify words that constitute a given fraction of the probability mass
    """
    nuc_ids = np.where(probs >= np.max(probs) * nuc_ratio)[0]
    nuc_pairs = sorted(zip(nuc_ids, probs[nuc_ids]), key = lambda x : -x[1])
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

def check_legal_english(x):
    if (x <= 670 or x >= 7992):
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
    cut_words = []
    # 防止进入死循环，比如，我把我把我把我把，遍历除了最后一个字符的整个句子，如果遇到与最后一个字符相等，就把下一个加入屏蔽词
    cut_words.extend([context[i+1] for i, word in enumerate(context[:-1]) if word == context[-1]]) # bigrams
    # 防止重复的名词
    # cut_words.extend([x for x in proposals if x not in STOPWORDS and in_context(x, context)]) # unigrams
    return [x for x in proposals if x not in cut_words]

class LanguageModel():
    """class for generating word sequences using a language model
    """

    def __init__(self, model, nuc_mass=1.0, nuc_ratio=0.0):
        self.model = model
        # self.ids = {i for word, i in self.model.word2id.items() if word in set(vocab)}
        self.nuc_mass, self.nuc_ratio = nuc_mass, nuc_ratio
        # with open("./vocab.json", "r", encoding='utf-8') as f:
        #     self.gpt_vocab = json.load(f)

    def ps(self, contexts):
        """get probability distributions over the next words for each context
        """
        context_arr = self.model.get_context_array(contexts)
        probs = self.model.get_probs(context_arr)
        return probs[:, len(contexts[0]) - 1]

    def beam_propose(self, beam, context_words):
        """get possible extension words for each hypothesis in the decoder beam
        """
        t1 = time.time()
        if len(beam) == 1:
            # nuc_words = [w for w in INIT if self.model.word2id[w] in self.ids]
            nuc_words = INIT
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

                nuc_ids = [i for i in nuc_ids if check_legal(i)]
                # nuc_ids = [i for i in nuc_ids if check_legal_english(i)]
                nuc_words = self.model.tokenizer.decode(nuc_ids)
                nuc_words = nuc_words.split(' ')
                # nuc_words = [i for i in nuc_words if d.check(i) and i.isalpha()]

                if nuc_words == [""]:
                    nuc_words = []
                nuc_words = context_filter(nuc_words, context)
                nuc_logprobs = np.log([probs[self.model.encode(w)[0]] for w in nuc_words])
                beam_nucs.append((nuc_words, nuc_logprobs))
            return beam_nucs

    def beam_propose_avoidnull(self, beam, context_words):
        """get possible extension words for each hypothesis in the decoder beam
        """
        t1 = time.time()
        if len(beam) == 1:
            # nuc_words = [w for w in INIT if self.model.word2id[w] in self.ids]
            nuc_words = INIT
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
                nuc_ids = [i for i in nuc_ids if check_legal(i)]
                nuc_words = self.model.tokenizer.decode(nuc_ids)
                nuc_words = nuc_words.split(' ')
                if nuc_words == [""]:
                    nuc_words = []
                nuc_words = context_filter(nuc_words, context)
                nuc_logprobs = np.log([probs[self.model.encode(w)[0]] for w in nuc_words])
                beam_nucs.append((nuc_words, nuc_logprobs))

            return beam_nucs