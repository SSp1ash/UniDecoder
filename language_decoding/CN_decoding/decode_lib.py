

import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats as ss
import re
from typing import List, Tuple, Optional

# ==================== 工具函数 ====================

def filter_chinese(text_list: List[str]) -> List[str]:

    pattern = re.compile(r'^[\u4e00-\u9fa5，。]+$')
    return [text for text in text_list if pattern.match(text)]


def filter_french(text_list: List[str]) -> List[str]:

    pattern = re.compile(r'^[a-zA-ZÀ-ÿ\s.,!?\'\"-]+$')
    return [text for text in text_list if pattern.match(text)]


def filter_NL(text_list: List[str], dictionary_path: str) -> List[str]:

    def load_dutch_dictionary(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip().lower() for line in f if line.strip())

    dutch_dict = load_dutch_dictionary(dictionary_path)
    return [word for word in text_list if word.strip().lower() in dutch_dict]


def get_nucleus(probs: np.ndarray, nuc_mass: float, nuc_ratio: float) -> List[int]:

    nuc_ids = np.where(probs >= np.max(probs) * nuc_ratio)[0]
    nuc_pairs = sorted(zip(nuc_ids, probs[nuc_ids]), key=lambda x: -x[1])
    nuc_pairs = nuc_pairs[:200]

    sum_mass = np.cumsum([x[1] for x in nuc_pairs])
    cutoffs = np.where(sum_mass >= nuc_mass)[0]
    if len(cutoffs) > 0:
        nuc_pairs = nuc_pairs[:cutoffs[0] + 1]
    return [x[0] for x in nuc_pairs]


def context_filter(proposals: List[str], context: List[str]) -> List[str]:

    cut_words = ['']
    cut_words.extend(context[-1])
    cut_words.extend([context[i + 1] for i, word in enumerate(context[:-1]) if word == context[-1]])
    return [x for x in proposals if x not in cut_words]



class Hypothesis:


    def __init__(self, parent=None, extension=None):
        if parent is None:
            self.words, self.logprobs = [], []
            self.likelihood = 0.0
            self.count = 0
        else:
            word, logprob, emb, likelihood = extension
            self.words = parent.words + [word]
            self.logprobs = parent.logprobs + [logprob]
            self.count = parent.count + 1

            if self.count % 5 == 0:
                self.likelihood = likelihood
            else:
                self.likelihood = parent.likelihood + likelihood




class Decoder:


    def __init__(self, word_times: np.ndarray, beam_width: int, extensions: int = 5):
        self.word_times = word_times
        self.beam_width = beam_width
        self.extensions = extensions
        self.beam = [Hypothesis()]
        self.scored_extensions = []

    def first_difference(self) -> int:

        words_arr = np.array([hypothesis.words for hypothesis in self.beam])
        if words_arr.shape[0] == 1:
            return words_arr.shape[1]
        for index in range(words_arr.shape[1]):
            if len(set(words_arr[:, index])) > 1:
                return index
        return 0

    def time_window(self, sample_index: int, seconds: float, floor: int = 0) -> int:

        window = [time for time in self.word_times if time < self.word_times[sample_index]
                  and time > self.word_times[sample_index] - seconds]
        return max(len(window), floor)

    def get_hypotheses(self) -> List[Tuple[Hypothesis, int]]:

        if len(self.beam[0].words) == 0:
            return zip(self.beam, [self.extensions for _ in self.beam])

        logprobs = [sum(hypothesis.logprobs) for hypothesis in self.beam]
        num_extensions = [int(np.ceil(self.extensions * rank / len(logprobs)))
                         for rank in ss.rankdata(logprobs)]
        return zip(self.beam, num_extensions)

    def add_extensions(self, extensions: List[Hypothesis], likelihoods: np.ndarray, num_extensions: int):

        scored_extensions = sorted(zip(extensions, likelihoods), key=lambda x: -x[0].likelihood)
        self.scored_extensions.extend(scored_extensions[:num_extensions])

    def extend(self, verbose: bool = False):

        self.beam = [x[0] for x in sorted(self.scored_extensions, key=lambda x: -x[0].likelihood)[:self.beam_width]]
        self.scored_extensions = []
        if verbose:
            print(self.beam[0].words)


# ==================== LLaMA3模型类 ====================

class LLaMA3:


    def __init__(self, model, tokenizer, device='cpu'):
        self.device = device
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.UNK_ID = 100

    def encode(self, words: str) -> List[int]:

        return self.tokenizer.encode(words, add_special_tokens=False)

    def get_context_array(self, contexts: List[List[str]]) -> torch.Tensor:

        context_array = [[self.encode(text) for text in inner_list] for inner_list in contexts]
        context_array = np.array(context_array)[:, :, 0]
        return torch.tensor(context_array).long().to(self.device)

    def get_hidden(self, ids: torch.Tensor, layer: int) -> np.ndarray:

        mask = torch.ones(ids.shape, dtype=torch.int8).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=ids.to(self.device),
                attention_mask=mask,
                output_hidden_states=True
            )

        target_hidden = outputs.hidden_states[layer][:, -1].detach().cpu().numpy()

        del outputs, mask
        torch.cuda.empty_cache()

        return target_hidden

    def get_probs(self, ids: torch.Tensor) -> np.ndarray:

        mask = torch.ones(ids.shape, dtype=torch.int8).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device), attention_mask=mask)

        probs = F.softmax(outputs.logits, dim=2).detach().cpu().numpy()

        del outputs, mask

        return probs


# ==================== 语言模型类 ====================

class LanguageModel:


    def __init__(self, model: LLaMA3, nuc_mass: float = 1.0, nuc_ratio: float = 0.0,
                 output_lan: str = 'CN', dictionary_path: Optional[str] = None):
        self.model = model
        self.nuc_mass = nuc_mass
        self.nuc_ratio = nuc_ratio
        self.output_lan = output_lan
        self.dictionary_path = dictionary_path


        if output_lan == 'CN':
            self.INIT = ['我', '你', '他', '她']
            prompt_text = '请在冒号后面继续生成文本，只能生成中文：'
        elif output_lan == 'EN':
            self.INIT = ['I', 'You', 'He', 'She', 'It', 'This', 'The']
            prompt_text = 'Please continue generating text after the colon, only English can be generated:'
        elif output_lan == 'FR':
            self.INIT = ['Je', 'Tu', 'Il', 'Elle', 'cela']
            prompt_text = 'Générer un texte basé sur ce qui suit les deux points, avec un contenu français:'
        elif output_lan == 'NL':
            self.INIT = ['Ik', 'Dat', 'De', 'Je', 'Ze', 'het', 'En']
            prompt_text = 'Ga verder met het genereren van tekst na de dubbele punt, alleen in het Nederlands:'


        self.prompt = self.model.tokenizer.encode(prompt_text, return_tensors='pt').to(self.model.device)

    def ps(self, contexts: List[List[str]]) -> np.ndarray:

        context_arr = self.model.get_context_array(contexts)
        full_input = torch.cat((self.prompt.repeat(context_arr.size(0), 1), context_arr), dim=1)
        probs = self.model.get_probs(full_input)

        del context_arr, full_input

        return probs[:, -1]

    def beam_propose(self, beam: List[Hypothesis], context_words: int) -> List[Tuple[List[str], np.ndarray]]:

        if len(beam) < 2:
            nuc_words = self.INIT
            nuc_logprobs = np.log(np.ones(len(nuc_words)) / len(nuc_words))
            return [(nuc_words, nuc_logprobs)]

        contexts = [hyp.words[-context_words:] for hyp in beam]
        beam_probs = self.ps(contexts)
        beam_nucs = []

        for context, probs in zip(contexts, beam_probs):
            nuc_ids = get_nucleus(probs, nuc_mass=self.nuc_mass, nuc_ratio=self.nuc_ratio)
            nuc_words = [self.model.tokenizer.decode([token_id], skip_special_tokens=True)
                        for token_id in nuc_ids]

            # 语言过滤
            if self.output_lan == 'CN':
                nuc_words = filter_chinese(nuc_words)
            elif self.output_lan == 'FR':
                nuc_words = filter_french(nuc_words)
            elif self.output_lan == 'NL':
                nuc_words = filter_NL(nuc_words, self.dictionary_path)

            if nuc_words == [""]:
                nuc_words = []

            nuc_words = context_filter(nuc_words, context)
            nuc_logprobs = np.log([probs[self.model.encode(w)[0]] for w in nuc_words])
            beam_nucs.append((nuc_words, nuc_logprobs))

        return beam_nucs




class LMFeatures:


    def __init__(self, model: LLaMA3, tokenizer, layer: int, context_words: int, batch_size: int = 20):
        self.model = model
        self.layer = layer
        self.context_words = context_words
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def extend(self, extensions: List[List[str]], verbose: bool = False) -> np.ndarray:

        contexts = [extension[-(self.context_words + 1):] for extension in extensions]

        if verbose:
            print(contexts)

        all_embs = []
        for i in range(0, len(contexts), self.batch_size):
            batch_contexts = contexts[i:i + self.batch_size]
            context_array = self.model.get_context_array(batch_contexts)
            batch_embs = self.model.get_hidden(context_array, layer=self.layer)
            all_embs.append(batch_embs)

            del context_array, batch_embs

        result = np.vstack(all_embs) if len(all_embs) > 1 else all_embs[0]
        return result




@torch.no_grad()
def contra_eb_char_level(extend_embs: np.ndarray, gt: torch.Tensor, indexs: int, device: str) -> np.ndarray:

    extend_embs = np.stack(extend_embs)
    extend_embs = torch.from_numpy(extend_embs).to(device)

    cos_sim = F.cosine_similarity(extend_embs, gt[:, indexs])
    like = cos_sim.cpu().numpy()

    del extend_embs, cos_sim

    return like


def decode_text_from_embeddings(
    referen_eb: torch.Tensor,
    model,
    tokenizer,
    device: str,
    output_lang: str = 'CN',
    feature_layer: int = 20,
    dictionary_path: Optional[str] = None,
    verbose: bool = True
) -> np.ndarray:

    length = referen_eb.shape[1]
    llama3 = LLaMA3(model, tokenizer, device)

    features = LMFeatures(model=llama3, tokenizer=tokenizer, layer=feature_layer,
                         context_words=15, batch_size=20)
    lm = LanguageModel(llama3, nuc_mass=0.9, nuc_ratio=0.03, output_lan=output_lang,
                      dictionary_path=dictionary_path)

    word_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    decoder = Decoder(word_times, 15)
    result = np.array([])

    try:
        for sample_index in range(length):
            if verbose and sample_index % 10 == 0:
                mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                print(f"[{sample_index}/{length}] GPU memory: {mem_allocated:.2f}GB")

            ncontext = 15
            beam_nucs = lm.beam_propose(decoder.beam, ncontext)

            for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
                nuc, logprobs = beam_nucs[c]

                if len(nuc) < 1:
                    continue

                extend_words = [hyp.words + [x] for x in nuc]
                extend_embs = list(features.extend(extend_words))
                likelihoods = contra_eb_char_level(extend_embs, referen_eb, sample_index, device)

                local_extensions = [Hypothesis(parent=hyp, extension=x) for x in
                                  zip(nuc, logprobs, [None] * len(nuc), likelihoods)]

                del extend_embs, extend_words
                decoder.add_extensions(local_extensions, likelihoods, nextensions)

            decoder.extend(verbose=False)
            result = np.array(decoder.beam[0].words)

            if sample_index % 20 == 0:
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error at step {sample_index}: {e}")
        torch.cuda.empty_cache()
        return result

    if verbose:
        print(f"✅ Decoding completed.")

    return result