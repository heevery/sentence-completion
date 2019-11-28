import os

from data_utils import settings, ProblemSet
from tokenization import save_vocab, Vocabulary, BaseTokenizer

if __name__ == "__main__":
    setname = 'msr'
    emb_dir = os.path.join(settings['lm1b_dir'], setname + '_embeddings')

    special_tokens = ['</S>', '<S>', '<UNK>']
    vocab = Vocabulary(special_tokens=special_tokens)
    tokenizer = BaseTokenizer(vocab, False, '<UNK>', special_tokens)

    problem_set = ProblemSet.load(setname)
    examples = problem_set.get_examples()

    for e in examples:
        for segment in e.context:
            vocab.update_counter(tokenizer.tokenize(segment))
        for candidate in e.candidates:
            for segment in candidate:
                vocab.update_counter(tokenizer.tokenize(segment))

    vocab.build_from_counter()

    if not os.path.exists(emb_dir):
        print("Creating directory at {}".format(emb_dir))
        os.makedirs(emb_dir)

    vocab_path = os.path.join(emb_dir, 'vocab.txt')
    print("Saving vocab at {} of size {}".format(vocab_path, len(vocab)))
    save_vocab(vocab, vocab_path)
