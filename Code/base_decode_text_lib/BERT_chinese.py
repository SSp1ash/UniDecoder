import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AlbertTokenizer, BertModel
from torch.nn.functional import softmax


class BERT():

    def __init__(self, path, device='cpu'):
        self.device = device
        self.model = BertModel.from_pretrained(path).eval().to(self.device)
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        # self.vocab = vocab
        # self.word2id = {w: i for i, w in enumerate(self.vocab)}
        # self.UNK_ID = self.tokenizer.decode('<unk>')
        self.UNK_ID = 100

    def encode(self, words):
        """map from words to ids
        """
        return self.tokenizer.encode(words,add_special_tokens=False)
        # return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]


    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i + nctx]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        # context_array = np.array([self.encode(words) for words in contexts])
        context_array = [[self.encode(text) for text in inner_list] for inner_list in contexts]
        context_array = np.array(context_array)[...,0]
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device),
                                 attention_mask=mask.to(self.device), output_hidden_states=True)
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        probs = softmax(outputs.logits, dim=2).detach().cpu().numpy()
        return probs