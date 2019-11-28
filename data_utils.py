import os
import re
import logging
import json
from typing import List
from ast import literal_eval

from tqdm import tqdm
import numpy as np
import pandas as pd

with open('settings.json', 'r') as f:
    settings = json.load(f)

logger = logging.getLogger(__name__)


class SentenceCompletionExample(object):
    def __init__(self, no, context, candidates, label):
        self.no = no
        self.context = context
        self.candidates = candidates
        self.label = label
        self.scores = None
        self.data = None

    def fill(self, i):
        chunks = [self.context[0]]
        for j, chunk in enumerate(self.candidates[i]):
            chunks.append(chunk)
            chunks.append(self.context[j + 1])
        return ''.join(chunks)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        attrs = [self.no, self.label, self.context, self.candidates]
        return "{} {} {} {}".format(*attrs)


def save_preds(examples: List[SentenceCompletionExample], path):
    with open(path, mode='w') as f:
        for e in examples:
            if e.scores is not None:
                row = [str(e.no)] + np.array(e.scores, dtype=str).tolist()
                f.write(','.join(row) + '\n')


class ProblemSet(object):
    def __init__(self, num_choices=2):
        self.num_choices = num_choices
        self.examples = []
        self.df = None

    def get_examples(self, partition=None) -> List[SentenceCompletionExample]:
        return self.examples if partition is None or partition == 'all' else []

    @classmethod
    def load(cls, name: str) -> 'ProblemSet':
        set_dir = settings['prob_set_dir']
        if name.lower() == 'msr':
            return MSR(os.path.join(set_dir, 'testing_data.csv'),
                       os.path.join(set_dir, 'test_answer.csv'))
        elif name.lower() == 'sat':
            return SAT(os.path.join(set_dir, 'SAT_completion.csv'))
        elif name.lower() == 'topik':
            return TOPIK(os.path.join(set_dir, 'topik_sample.csv'))
        else:
            return ProblemSet()


class MSR(ProblemSet):
    def __init__(self, problem_fp, answer_fp):
        super(MSR, self).__init__(5)
        prob = pd.read_csv(problem_fp)
        ans = pd.read_csv(answer_fp)
        self.df = pd.merge(prob, ans, left_on='id', right_on='id')

        for key, row in self.df.iterrows():
            context = re.split('_+', row['question'])
            cs = [[c] for c in row[['a)', 'b)', 'c)', 'd)', 'e)']]]
            label = ord(row['answer']) - 97
            e = SentenceCompletionExample(row['id'], context, cs, label)
            self.examples.append(e)

    def get_examples(self, partition=None) -> List[SentenceCompletionExample]:
        if partition in ("valid", "va"):
            return self.examples[:len(self.examples) // 2]
        elif partition in ("test", "te"):
            return self.examples[len(self.examples) // 2:]
        else:
            return super().get_examples(partition)


class SAT(ProblemSet):
    def __init__(self, file_path):
        super(SAT, self).__init__(5)
        cvt = {'{})'.format(chr(i + 97)): literal_eval for i in range(5)}
        self.df = pd.read_csv(file_path, converters=cvt)

        for key, row in self.df.iterrows():
            context = re.split('_+', row['question'])
            cs = [c for c in row[['a)', 'b)', 'c)', 'd)', 'e)']]]
            label = ord(row['answer']) - 97
            e = SentenceCompletionExample(row['id'], context, cs, label)
            self.examples.append(e)

    def get_examples(self, partition=None) -> List[SentenceCompletionExample]:
        if partition in ("test", "te"):
            return self.examples
        return super().get_examples(partition)


class TOPIK(ProblemSet):
    def __init__(self, file_path):
        super(TOPIK, self).__init__(4)
        self.df = pd.read_csv(file_path)
        choice_names = ['C%d' % i for i in range(1, 5)]
        self.df.columns = ['No', 'Ro', 'Lv', 'Part', 'Type', 'PN', 'A', 'Q'] \
                          + choice_names

        prev_psg = ''
        psg_id = -1
        psg_ids = []
        for key, row in self.df.iterrows():
            context = re.split('\(\)', row['Q'])
            cs = [[c] for c in row[choice_names]]
            e = SentenceCompletionExample(row['No'], context, cs, row['A'] - 1)
            self.examples.append(e)

            psg = e.fill(e.label)
            if psg != prev_psg:
                psg_id += 1
                prev_psg = psg
            psg_ids.append(psg_id)
        self.df['psg_id'] = psg_ids

    def get_examples(self, partition=None) -> List[SentenceCompletionExample]:
        if partition in ("valid", "va"):
            s = (self.df['psg_id'] % 2 == 0)
            return [e for i, e in enumerate(self.examples) if s[i]]
        elif partition in ("test", "te"):
            s = (self.df['psg_id'] % 2 == 1)
            return [e for i, e in enumerate(self.examples) if s[i]]
        else:
            return super().get_examples(partition)


class LineInput(object):
    def __init__(self, txt_paths, tokenizer, update, min_len=1, max_len=120):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        if update:
            for word_tokens in self.tokenize_txt(txt_paths, min_len, max_len):
                self.vocab.update_counter(word_tokens)

        self.txt_paths = txt_paths
        self.min_len = min_len
        self.max_len = max_len
        self.data = None

    def load_data(self):
        len2lines = dict()
        for tokens in self.tokenize_txt(self.txt_paths, self.min_len, self.max_len):
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            lines = len2lines.get(len(ids), [])
            lines.append(ids)
            len2lines[len(ids)] = lines
        return len2lines

    def tokenize_txt(self, txt_paths, min_len, max_len):
        for txt_path in tqdm(txt_paths, desc="Files"):
            with open(txt_path, 'r') as f:
                # assume that each line is a sentence
                for line in f.readlines():
                    tokens = self.tokenizer.tokenize(line)
                    if min_len <= len(tokens) <= max_len:
                        yield tokens

    def batchify(self, bsz, shuffle=False):
        if self.data is None:
            self.data = self.load_data()

        batches = []
        n_lines, remainder = 0, 0
        for len_, lines in self.data.items():
            n_lines += len(lines)
            if shuffle:
                np.random.shuffle(lines)
            nb = len(lines) // bsz
            remainder += len(lines) % bsz
            if nb > 0:
                batches.extend([lines[i*bsz:(i+1)*bsz] for i in range(nb)])

        if shuffle:
            np.random.shuffle(batches)
        logger.debug("Discarded {} of {} lines resulting in {} batches"
                     .format(remainder, n_lines, len(batches)))
        return batches


def get_txts(name, mode):
    return [os.path.join(settings['prepro_dir'], name, '{}.txt'.format(mode))]
