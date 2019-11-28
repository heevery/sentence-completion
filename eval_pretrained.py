import os
import os.path as osp
import glob
import argparse
import json
import logging
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_transformers as pt
from pytorch_transformers import (
    BertConfig, BertForPreTraining, BertTokenizer,
    OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    WEIGHTS_NAME, CONFIG_NAME)

from model import WordRNN, LM1B
from data_utils import settings
from data_utils import ProblemSet, SentenceCompletionExample, save_preds
from tokenization import load_vocab, BaseTokenizer, NLTKTokenizer

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForPreTraining, BertTokenizer),
    'gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}
MODEL_CLASSES['openai-gpt'] = MODEL_CLASSES['gpt']


def evaluate(examples: List[SentenceCompletionExample], model: nn.Module,
             tokenizer, direction, criterion, name='model'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    for e in examples:
        context_tokens = [tokenizer.tokenize(t) for t in e.context]
        e.scores = []
        for i in range(len(e.candidates)):
            candidate_tokens = [tokenizer.tokenize(c) for c in e.candidates[i]]
            tokens = list(context_tokens[0])
            blank_mask = [0] * len(context_tokens[0])
            for j in range(len(candidate_tokens)):
                tokens.extend(candidate_tokens[j])
                tokens.extend(context_tokens[j + 1])
                blank_mask += [1] * len(candidate_tokens[j])
                blank_mask += [0] * len(context_tokens[j + 1])

            ids = tokenizer.convert_tokens_to_ids(tokens)
            ids = torch.tensor(ids, dtype=torch.long).to(device)
            blank_mask = torch.tensor(blank_mask, dtype=torch.uint8).to(device)

            if direction == 'forward':
                in_ids = ids[:-1].unsqueeze(0)
                out_ids = ids[1:]
                out_mask = blank_mask[1:]
            elif direction == 'bidirec':
                in_ids = ids.unsqueeze(0)
                out_ids = ids[1:-1]
                out_mask = blank_mask[1:-1]
            elif direction == 'autoenc':
                mask_id = tokenizer.vocab['[MASK]']
                in_ids = ids.masked_fill(blank_mask, mask_id).unsqueeze(0)
                out_ids = ids
                out_mask = blank_mask

            if criterion == 'blank':
                out_ids = out_ids.masked_fill(1 - out_mask, -1)
            elif criterion == 'partial':
                out_ids = out_ids.masked_fill(out_mask, -1)

            with torch.no_grad():
                logits = model(in_ids)[0]
                logits_flat = logits.view(-1, model.config.vocab_size)
                loss = F.cross_entropy(logits_flat, out_ids,
                                       reduction='sum', ignore_index=-1)
                e.scores.append(-loss.item())
        logger.debug("No. {}, Prediction: {}, Answer: {}".format(
                e.no, np.argmax(e.scores), e.label))

    correct = [np.argmax(e.scores) == e.label for e in examples]
    accuracy = np.mean(correct) * 100
    logger.info("Accuracy of {}: {:4.2f}%".format(name, accuracy))


def move_cached(name, cache_dir, out_path):
    cached_vocab = pt.cached_path(name, cache_dir=cache_dir)
    logger.info("Moving cached vocab {} to {}".format(
            cached_vocab, out_path))
    os.rename(cached_vocab, out_path)
    os.remove(cached_vocab + '.json')


def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model', type=str, default='wordrnn')
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='nltk',
                        help='Only effective when model set to wordrnn')
    parser.add_argument('--criterion', type=str, default='full')

    # data
    parser.add_argument('--set', type=str, default='msr')
    parser.add_argument('--partition', type=str, default='va')
    parser.add_argument('--no-move-cached', action='store_true')

    parser.add_argument('--log-dir', type=str, default='train/noname')
    parser.add_argument('--save-pred', action='store_true')

    args = parser.parse_args()

    problem_set = ProblemSet.load(args.set)
    examples = problem_set.get_examples(args.partition)

    logger.info("Evaluating models saved in {} on {}-{}".format(
            args.dir, args.set, args.partition))

    if not os.path.exists(args.log_dir):
        logger.info("Creating directory at {}".format(args.log_dir))
        os.makedirs(args.log_dir)

    args_path = os.path.join(args.log_dir, 'args.json')
    with open(args_path, 'w') as f:
        logger.info("Saving arguments at {}".format(args_path))
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(args.log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    model_type = args.model.lower()
    if model_type == 'wordrnn':
        args_path = osp.join(args.dir, 'args.json')
        with open(args_path, 'r') as f:
           arg_dict = json.load(f)

        vocab_path = osp.join(args.dir, 'vocab.txt')
        vocab = load_vocab(vocab_path)
        if args.tokenizer.lower() == 'nltk':
            tokenizer = NLTKTokenizer(vocab, arg_dict['lower'])
        elif args.tokenizer.lower() == 'wordpiece':
            tokenizer = BertTokenizer(vocab_path, arg_dict['lower'])
        model = WordRNN(
                len(vocab), len(vocab), arg_dict['rnncell'],
                arg_dict['emsize'], arg_dict['outsize'], arg_dict['nhid'],
                arg_dict['nlayers'], arg_dict['bidirec'],
                arg_dict.get('autoenc', False), arg_dict['decoder_bias'])
        logger.info(model)

        ckpt_paths = glob.glob(osp.join(args.dir, '*.pt'))
        ckpt_paths.sort(key=osp.getmtime)
        for path in ckpt_paths:
            model.load_state_dict(torch.load(path))
            direction = 'autoenc' if model.autoenc else (
                'bidirec' if model.bidirec else 'forward')
            evaluate(examples, model, tokenizer, direction, args.criterion,
                     str(osp.basename(path.split('.')[0])))
            if args.save_pred:
                save_fn = osp.basename(path).replace('.pt', '.csv')
                save_preds(examples, osp.join(args.log_dir, save_fn))
    elif model_type == 'lm1b':
        lm1b_dir = settings['lm1b_dir']

        for e in examples:
            e.context[0] = ' '.join(['<S>', e.context[0]])
            e.context[-1] = ' '.join([e.context[-1], '</S>'])

        vocab = load_vocab(osp.join(lm1b_dir, 'vocab-2016-09-10.txt'))
        special_tokens = ['<S>', '</S>', '<UNK>']
        tokenizer = BaseTokenizer(vocab, False, '<UNK>', special_tokens)
        in_vocab = load_vocab(osp.join(lm1b_dir, args.dir, 'vocab.txt'))

        out_to_in = [in_vocab['<UNK>']] * 800000
        for i, token in tokenizer.ids_to_tokens.items():
            out_to_in[i] = in_vocab.get(token, in_vocab['<UNK>'])

        tf_path = osp.join(lm1b_dir, 'ckpt-*')
        npy_path = osp.join(lm1b_dir, args.dir, 'embeddings.npy')
        model = LM1B.from_tf(tf_path, npy_path, out_to_in, 8)
        logger.info(model)

        evaluate(examples, model, tokenizer, 'forward', args.criterion)
        if args.save_pred:
            save_preds(examples, osp.join(args.log_dir, 'preds.csv'))
    else:
        cache_dir = settings['pretrans_dir']
        bert_dir = osp.join(settings['pretrans_dir'], args.dir)
        model_or_dir = bert_dir if osp.exists(bert_dir) else args.dir

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(
                model_or_dir, cache_dir=cache_dir)
        tokenizer = tokenizer_class.from_pretrained(
                model_or_dir, cache_dir=cache_dir,
                max_len=config.max_position_embeddings,
                do_lower_case='-uncased' in model_or_dir)
        model = model_class.from_pretrained(
                model_or_dir, cache_dir=cache_dir, config=config)

        direction = 'forward'
        if model_type == 'bert':
            direction = 'autoenc'

        evaluate(examples, model, tokenizer, direction, args.criterion)
        if args.save_pred:
            save_preds(examples, osp.join(args.log_dir, 'preds.csv'))

        if not args.no_move_cached and not osp.exists(bert_dir):
            logger.info("Creating directory at {}".format(bert_dir))
            os.mkdir(bert_dir)

            model_url = model.pretrained_model_archive_map[model_or_dir]
            model_path = osp.join(bert_dir, WEIGHTS_NAME)
            move_cached(model_url, cache_dir, model_path)

            config_url = model.config.pretrained_config_archive_map[model_or_dir]
            config_path = osp.join(bert_dir, CONFIG_NAME)
            move_cached(config_url, cache_dir, config_path)

            for k, url_map in tokenizer.pretrained_vocab_files_map.items():
                vocab_path = osp.join(bert_dir, tokenizer.vocab_files_names[k])
                move_cached(url_map[model_or_dir], cache_dir, vocab_path)


if __name__ == "__main__":
    main()
