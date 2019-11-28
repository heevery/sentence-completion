import os
import logging
from collections import UserDict, Counter, OrderedDict

from nltk import tokenize

from pytorch_transformers.tokenization_bert import (
    load_vocab, whitespace_tokenize, BasicTokenizer)

logger = logging.getLogger(__name__)


class Vocabulary(UserDict):
    def __init__(self, special_tokens=(
            '[UNK]', '[SEP]', '[CLS]', '[PAD]', '[MASK]')):
        super().__init__(self)
        self.counter = Counter()
        self.words = []

        for token in special_tokens:
            self.add_word(token)

    def update_counter(self, words):
        self.counter.update(words)

    def add_word(self, word):
        if word not in self.data:
            self.data[word] = len(self.words)
            self.words.append(word)
        return self.data[word]

    def build_from_counter(self, min_cnt=1):
        # tie break with alphabetical order
        for w, cnt in sorted(self.counter.items(),
                             key=lambda x: (-x[1], x[0])):
            if cnt >= max(min_cnt, 1):
                self.add_word(w)


def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as writer:
        index = 0
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            if index != token_index:
                logger.warning("Saving vocab: index jumps from {} to {}"
                               .format(index, token_index))
                index = token_index
            writer.write(token + u'\n')
            index += 1


class Tokenizer(object):
    def __init__(self, vocab, do_lower_case=True, unk_token='[UNK]'):
        if isinstance(vocab, str) and os.path.isfile(vocab):
            self.vocab = load_vocab(vocab)
        else:
            self.vocab = vocab
        self.ids_to_tokens = OrderedDict(
                [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.unk_id = self.vocab.get(unk_token, -1)

    def tokenize(self, text):
        return whitespace_tokenize(text.lower())

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids_to_tokens[i] for i in ids]


class BaseTokenizer(Tokenizer):
    def __init__(self, vocab, do_lower_case=True, unk_token='[UNK]',
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        super(BaseTokenizer, self).__init__(vocab, do_lower_case, unk_token)
        self.never_split = list(never_split)
        self._tokenizer = BasicTokenizer(do_lower_case, self.never_split)

    def tokenize(self, text):
        return self._tokenizer.tokenize(text)


class NLTKTokenizer(Tokenizer):
    def tokenize(self, text):
        tokens = tokenize.word_tokenize(text)
        return [token.lower() for token in tokens]
