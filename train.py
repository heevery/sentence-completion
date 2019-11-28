import os
import time
import math
import logging
import argparse
import json

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from tokenization import load_vocab, save_vocab, Vocabulary, Tokenizer
from data_utils import get_txts, LineInput
from model import WordRNN


logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_xy(model, source, device):
    source_t = torch.tensor(source, dtype=torch.long, device=device)
    if model.autoenc:
        x, mask = add_noise(source_t, 0.15, model.in_tokens)
        y = torch.full(x.size(), -100, dtype=torch.long, device=device)
        y[mask] = source_t[mask]
    else:
        x = source_t if model.bidirec else source_t[:, :-1]
        y = source_t[:, 1:-1] if model.bidirec else source_t[:, 1:]
    return x, y.reshape(-1)


def add_noise(tensor, rate, ntokens, mask_id=4, ratio=(.8, .1, .1)):
    copied = tensor.clone()
    rand_matrix = torch.rand(copied.size())
    rand_token_ids = torch.randint_like(copied, ntokens)
    reconstruction_mask = rand_matrix < rate

    copied[rand_matrix < rate * ratio[0]] = mask_id
    random_mask = ((rand_matrix > rate * ratio[0]) *
                   (rand_matrix < rate * (ratio[0] + ratio[1])))
    copied[random_mask] = rand_token_ids[random_mask]
    return copied, reconstruction_mask


def evaluate(model: nn.Module, batches, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss, total_len = 0, 0
    with torch.no_grad():
        for batch_data in batches:
            x, y = get_xy(model, batch_data, device)

            logits, __ = model(x)
            logits_flat = logits.view(-1, model.out_tokens)
            loss = F.cross_entropy(logits_flat, y, reduction='sum')

            total_loss += loss.item()
            total_len += (y >= 0).sum().item()
    return total_loss / total_len


def train(model: nn.Module, batches, learnables, optimizer: optim.Optimizer,
          device, args: argparse.Namespace):
    # Turn on training mode which enables dropout.
    model.train()

    total_loss, total_len = 0, 0
    start_time = time.time()
    for i, batch_data in enumerate(batches, start=1):
        x, y = get_xy(model, batch_data, device)

        model.zero_grad()
        logits, __ = model(x)
        logits_flat = logits.view(-1, model.out_tokens)
        loss = F.cross_entropy(logits_flat, y, reduction='sum')

        total_loss += loss.item()
        total_len += (y >= 0).sum().item()

        loss /= len(batch_data)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(learnables, args.clip)
        optimizer.step()

        remainder = i % args.log_interval
        if remainder == 0 or i == len(batches):
            cur_loss = total_loss / total_len
            denom = (args.log_interval if remainder == 0 else remainder)
            time_per_batch = (time.time() - start_time) * 1000 / denom
            logger.info('| {:6d} bs | ms/b {:5.2f} '
                        '| loss {:5.2f} | ppl {:8.2f} |'
                        .format(i, time_per_batch,
                                cur_loss, math.exp(cur_loss)))
            total_loss, total_len = 0, 0
            start_time = time.time()


def main():
    parser = argparse.ArgumentParser()

    # model structure
    parser.add_argument('--rnncell', type=str, default='LSTM')
    parser.add_argument('--emsize', type=int, default=200)
    parser.add_argument('--nhid', type=int, default=600)
    parser.add_argument('--outsize', type=int, default=400)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--bidirec', action='store_true')
    parser.add_argument('--autoenc', action='store_true')
    parser.add_argument('--forget-bias', type=float, default=False)
    parser.add_argument('--decoder-bias', action='store_true')

    # data
    parser.add_argument('--corpus', type=str, default='guten')
    parser.add_argument('--min-len', type=int, default=10)
    parser.add_argument('--max-len', type=int, default=80)

    # vocabulary
    parser.add_argument('--vocab', type=str, default=None)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--min-cnt', type=int, default=6)

    # training
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=3333)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--eval-batch-size', type=int, default=10)

    # optimizer
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=.5)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--decay-after', type=int, default=5)
    parser.add_argument('--decay-rate', type=float, default=0.5)
    parser.add_argument('--decay-period', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)

    # save and log
    parser.add_argument('--save-dir', type=str, default='train/noname')
    parser.add_argument('--log-interval', type=int, default=10000)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--save-all', action='store_false')
    parser.add_argument('--save-period', type=int, default=1)

    args = parser.parse_args()
    logger.debug("Running {}".format(__file__))

    if not os.path.exists(args.save_dir):
        logger.debug("Creating directory at {}".format(args.save_dir))
        os.makedirs(args.save_dir)

    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as f:
        logger.debug("Saving arguments at {}".format(args_path))
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(args.save_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use pre built vocabulary if it exists
    if args.vocab and os.path.exists(args.vocab):
        vocab = load_vocab(args.vocab)
        update = False
    else:
        vocab = Vocabulary()
        update = True
    tokenizer = Tokenizer(vocab, args.lower)

    tr_txts = get_txts(args.corpus, 'train')
    va_txts = get_txts(args.corpus, 'valid')

    tr_input = LineInput(
            tr_txts, tokenizer, update, args.min_len, args.max_len)
    va_input = LineInput(
            va_txts, tokenizer, update, args.min_len, args.max_len)
    va_batches = va_input.batchify(args.eval_batch_size, False)

    if update:
        vocab.build_from_counter(args.min_cnt)
        logger.debug("Built vocab of size {}".format(len(vocab)))

    # Build the model
    model = WordRNN(len(vocab), len(vocab), args.rnncell,
                    args.emsize, args.outsize, args.nhid, args.nlayers,
                    args.bidirec, args.autoenc, args.decoder_bias,
                    args.forget_bias, args.dropout)
    logger.debug(model)
    model.to(device)

    learnables = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = getattr(optim, args.optim)(learnables, lr=args.lr)

    save_vocab(vocab, os.path.join(args.save_dir, 'vocab.txt'))
    model_path = os.path.join(args.save_dir, 'model.pt')

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # Loop over epochs.
        best_val_loss = None

        logger.info('-' * 79)
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            tr_batches = tr_input.batchify(args.batch_size, True)
            train(model, tr_batches, learnables, optimizer, device, args)

            val_loss = evaluate(model, va_batches, device)
            logger.info('-' * 79)
            logger.info('| end of epoch {:2d} | time: {:5.2f}s '
                        '| valid loss {:5.2f} | valid ppl {:8.2f} |'
                        .format(epoch, (time.time() - epoch_start_time),
                                val_loss, math.exp(val_loss)))
            logger.info('-' * 79)

            updated_best = not best_val_loss or val_loss < best_val_loss
            if epoch >= args.decay_after > 0:
                if (epoch - args.decay_after) % args.decay_period == 0:
                    for group in optimizer.param_groups:
                        group['lr'] *= args.decay_rate

            if (epoch % args.save_period == 0) and (updated_best or args.save_all):
                if args.save_all:
                    model_path = os.path.join(args.save_dir, 'ep{}.pt'.format(epoch))
                torch.save(model.state_dict(), model_path)

                if updated_best:
                    best_val_loss = val_loss

        logger.debug("Completed training and saved to {}".format(args.save_dir))
    except KeyboardInterrupt:
        logger.debug('-' * 79)
        logger.debug("Exiting from training early")


if __name__ == "__main__":
    main()
