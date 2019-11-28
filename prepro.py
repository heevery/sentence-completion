import os
from tqdm import tqdm
import pandas as pd

from data_utils import settings
from tokenization import Vocabulary, save_vocab
from pytorch_transformers import BertTokenizer


def preprocess_sejong():
    dir_ = os.path.join(settings['bert_dir'], 'bert-base-multilingual-cased')
    tokenizer = BertTokenizer.from_pretrained(dir_, do_lower_case=False)
    vocab = Vocabulary()

    src_path = os.path.join(settings['sejong_dir'], 'sejong')
    tr_path = os.path.join(settings['prepro_dir'], 'sejong', 'train.txt')
    va_path = os.path.join(settings['prepro_dir'], 'sejong', 'valid.txt')

    lengths = []
    with open(src_path, 'r') as sf, \
            open(tr_path, 'w') as tf, open(va_path, 'w') as vf:
        for i, line in enumerate(tqdm(sf.readlines(-1)), 1):
            tokens = tokenizer.tokenize(line)
            lengths.append(len(tokens))
            vocab.update_counter(tokens)

            if i % 100 == 0:
                vf.write(' '.join(tokens) + "\n")
            else:
                tf.write(' '.join(tokens) + "\n")

    vocab.build_from_counter()
    # save_vocab(vocab, "train/sejong/wordpiece.txt")

    prev_cnt = -1
    for w, cnt in sorted(vocab.counter.items(), key=lambda x: (-x[1], x[0])):
        if cnt == prev_cnt:
            continue
        print("Count: {}, Index: {}, Token: {}".format(
                cnt, vocab.get(w, -1), w))
        prev_cnt = cnt
    print("Built vocab of size {}".format(len(vocab)))

    np = 20
    percentiles = [i / np for i in range(1, np)] + [0.99]
    with pd.option_context('precision', 1):
        print(pd.Series(lengths).describe(percentiles))


if __name__ == "__main__":
    preprocess_sejong()
