import os
import argparse
import logging
import json
from typing import List

from tqdm import tqdm
import numpy as np
import torch
from torch import nn

from pytorch_transformers import (
    BertConfig, BertForMultipleChoice, BertTokenizer,
    BertForSequenceClassification,
    OpenAIGPTConfig, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
    GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer,
    AdamW, WarmupLinearSchedule)

from data_utils import settings
from data_utils import ProblemSet, SentenceCompletionExample, save_preds

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'gpt': (OpenAIGPTConfig, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer),
    'gpt2': (GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer),
}
MODEL_CLASSES['openai-gpt'] = MODEL_CLASSES['gpt']


def example_to_feature(example: SentenceCompletionExample, tokenizer,
                       cls_token_at_end=False, cls_token='[CLS]',
                       sep_token='[SEP]', bos_token=None):
    choice_features = []
    context_tokens = [tokenizer.tokenize(t) for t in example.context]

    max_seq_len = -1
    for i in range(len(example.candidates)):
        candidate_tokens = [tokenizer.tokenize(c) for c in
                            example.candidates[i]]
        tokens = []
        if not cls_token_at_end:
            tokens.append(cls_token)
        if bos_token:
            tokens.append(bos_token)

        tokens.extend(context_tokens[0])
        segment_ids = [0] * len(tokens)

        for j in range(len(candidate_tokens)):
            if sep_token:
                tokens.append(sep_token)
                segment_ids.append(0)

            tokens.extend(candidate_tokens[j])
            segment_ids.extend([1] * len(candidate_tokens[j]))

            if sep_token:
                tokens.append(sep_token)
                segment_ids.append(1)

            tokens.extend(context_tokens[j + 1])
            segment_ids.extend([0] * len(context_tokens[j + 1]))

        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(0)
        elif sep_token:
            tokens.append(sep_token)
            segment_ids.append(0)

        assert len(tokens) == len(segment_ids)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        choice_features.append((input_ids, segment_ids))

        if max_seq_len < 0 or len(input_ids) > max_seq_len:
            max_seq_len = len(input_ids)

    return _pad(choice_features, max_seq_len, cls_token_at_end)


def _pad(choice_features, max_seq_len, cls_token_at_end):
    input_ids = []
    input_mask = []
    segment_ids = []
    lm_labels = []
    mc_token_ids = []
    for (choice_input_ids, choice_segment_ids) in choice_features:
        seq_len = len(choice_input_ids)
        pad_len = max_seq_len - seq_len

        input_ids.append(choice_input_ids + [0] * pad_len)
        lm_labels.append(choice_input_ids + [-1] * pad_len)
        segment_ids.append(choice_segment_ids + [0] * pad_len)
        input_mask.append([1] * seq_len + [0] * pad_len)
        mc_token_ids.append((seq_len - 1) if cls_token_at_end else 0)

    return input_ids, input_mask, segment_ids, lm_labels, mc_token_ids


def get_gpt_xy(input_ids, mc_token_ids, lm_labels, label):
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    mc_token_ids = torch.tensor(mc_token_ids, dtype=torch.long).unsqueeze(0)
    lm_labels = torch.tensor(lm_labels, dtype=torch.long).unsqueeze(0)

    label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
    return input_ids, mc_token_ids, lm_labels, label


def get_xy(input_ids, input_mask, segment_ids, label, modeling_type):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    if segment_ids:
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

    if modeling_type in ('sequence-classification', 'seq-cls'):
        label_t = torch.zeros(input_ids.size(0), dtype=torch.long)
        label_t[label] = 1
        label = label_t
    else:
        input_ids.unsqueeze_(0)
        input_mask.unsqueeze_(0)
        if segment_ids is not None:
            segment_ids.unsqueeze_(0)
        label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
    return input_ids, segment_ids, input_mask, label


def evaluate(model: nn.Module, examples: List[SentenceCompletionExample],
             device, args):
    model.eval()
    with torch.no_grad():
        for example in examples:
            tensors = tuple((d if d is None else d.to(device))
                            for d in example.data[:-1])
            outputs = model(*tensors)
            logits = outputs[2] if 'gpt' in args.model.lower() else outputs[0]
            # hack to handle both formulation types
            example.scores = logits.t().flatten()[-tensors[0].size(-2):].cpu()

    correct = [np.argmax(e.scores) == e.label for e in examples]
    return np.mean(correct)


def finetune(model: nn.Module, examples: List[SentenceCompletionExample],
             optimizer, scheduler, device, args):
    model.train()
    total_loss, total_len = 0, 0
    for i, example in enumerate(tqdm(examples, desc="Iteration"), start=1):
        tensors = tuple((d if d is None else d.to(device))
                        for d in example.data)

        model.zero_grad()
        outputs = model(*tensors)
        loss = (args.lm_coef * outputs[0] + outputs[1]
                if 'gpt' in args.model.lower() else outputs[0])
        total_loss += loss.sum().item()
        total_len += len(tensors[-1])

        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / total_len


def main():
    parser = argparse.ArgumentParser()

    # pretrained model
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--pretrained', type=str,
                        default='bert-large-uncased-whole-word-masking')

    # data
    parser.add_argument('--tr-set', type=str, default='msr')
    parser.add_argument('--tr-partition', type=str, default='va')
    parser.add_argument('--holdout', type=float, default=100.)
    parser.add_argument('--ev-set', type=str, default=None)
    parser.add_argument('--ev-partition', type=str, default='va')

    # formulation
    parser.add_argument('--modeling-type', type=str, default='multi-choice')
    parser.add_argument('--sep', action='store_true')
    parser.add_argument('--bos', action='store_true')
    parser.add_argument('--segment', action='store_true')

    # optimization
    parser.add_argument('--update-embeddings', action='store_true')
    parser.add_argument('--fix-layer-to', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3333)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument("--warmup", type=float, default=1.)
    parser.add_argument("--lm-coef", type=float, default=.5)

    # save and log
    parser.add_argument('--eval-training', action='store_true')
    parser.add_argument('--log-dir', type=str, default='train/noname')

    args = parser.parse_args()
    logger.info("Finetuning model of {}".format(args.pretrained))

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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dataset setup
    problem_set = ProblemSet.load(args.tr_set)
    dev_examples = problem_set.get_examples(args.tr_partition)
    examples = list(dev_examples)
    if args.ev_set:
        ev_set = ProblemSet.load(args.ev_set)
        ev_examples = ev_set.get_examples(args.ev_partition)
        examples.extend(ev_examples)

    model_or_dir = os.path.join(settings['pretrans_dir'], args.pretrained)
    model_type = args.model.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_or_dir)

    tokenizer = tokenizer_class.from_pretrained(
            model_or_dir, max_len=config.max_position_embeddings,
            do_lower_case='-uncased' in model_or_dir)

    special_tokens = {'cls_token': '[CLS]'}
    if args.sep:
        special_tokens['sep_token'] = '[SEP]'
    if args.bos:
        special_tokens['bos_token'] = '[BOS]'
    if model_type == 'gpt2':
        setattr(tokenizer, 'unk_token', '<unk>')
    num_added = tokenizer.add_special_tokens(special_tokens)

    sep_token = special_tokens.get('sep_token', None)
    for e in examples:
        if 'gpt' in model_type:
            input_ids, __, __, lm_labels, mc_token_ids = example_to_feature(
                    e, tokenizer, True, sep_token=sep_token,
                    bos_token=special_tokens.get('bos_token', None))
            e.data = get_gpt_xy(input_ids, mc_token_ids, lm_labels, e.label)
        else:
            input_ids, input_mask, segment_ids, __, __ = example_to_feature(
                    e, tokenizer, False, sep_token=sep_token)
            if not args.segment:
                segment_ids = None
            e.data = get_xy(input_ids, input_mask, segment_ids, e.label,
                            args.modeling_type)

    ho = max(args.holdout, 0.)
    n_holdout = (int(ho) if ho >= 1. else int(len(examples) * ho))
    n_training = len(dev_examples) - n_holdout
    tr_examples = dev_examples[:n_training]
    ho_examples = dev_examples[n_training:]

    # load pre-trained model
    if args.modeling_type == 'multi-choice':
        model = model_class.from_pretrained(model_or_dir, config=config)
    elif model_type == 'bert' and args.modeling_type in (
            'sequence-classification', 'seq-cls'):
        config.num_labels = 2
        model = BertForSequenceClassification.from_pretrained(
                model_or_dir, config=config)
    model.resize_token_embeddings(config.vocab_size + num_added)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if n_holdout > 0:
        # test before fine-tuning
        accuracy = evaluate(model, ho_examples, device, args)
        logger.info("Holdout accuracy before fine-tuning: {:.2f}".format(
                accuracy * 100))

    # optimizer setup
    param_optimizer = list(model.named_parameters())

    no_update = []
    if not args.update_embeddings:
        no_update.extend(['embed', 'wte', 'wpe'])
    if args.fix_layer_to != 0:
        fix_layer_to = int(args.fix_layer_to)
        if fix_layer_to < 0:
            fix_layer_to = model.config.num_hidden_layers + fix_layer_to
        for i in range(fix_layer_to):
            no_update.append('layer.{}.'.format(i)) # for bert
            no_update.append('h.{}.'.format(i)) # for gpt and gpt2
    param_optimizer = [tup for tup in param_optimizer if
                       not any(nd in tup[0] for nd in no_update)]

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    t_total = len(tr_examples) * args.epochs
    warmup_steps = len(tr_examples) * args.warmup
    scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=t_total)

    # supervised fine-tuning
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(tr_examples)
        loss = finetune(model, tr_examples, optimizer, scheduler, device, args)
        logger.info("Fine-tuning loss for epoch {}: {:.4f}".format(
                epoch, loss))

        if args.eval_training:
            accuracy = evaluate(model, tr_examples, device, args)
            logger.info("Training accuracy after epoch {}: {:.2f}".format(
                    epoch, accuracy * 100))

        if n_holdout > 0:
            accuracy = evaluate(model, ho_examples, device, args)
            logger.info("Holdout accuracy after epoch {}: {:.2f}".format(
                    epoch, accuracy * 100))

        dev_path = os.path.join(args.log_dir, 'dev_ep{}.csv'.format(epoch))
        save_preds(dev_examples, dev_path)

        if args.ev_set:
            accuracy = evaluate(model, ev_examples, device, args)
            logger.info("{}-{} accuracy after epoch {}: {:.2f}".format(
                    args.ev_set, args.ev_partition, epoch, accuracy * 100))

            ev_path = os.path.join(args.log_dir, 'ev_ep{}.csv'.format(epoch))
            save_preds(ev_examples, ev_path)


if __name__ == "__main__":
    main()
