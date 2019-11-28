# Word-Level Neural LMs for Sentence Completion
A pytorch implementation of the assessment of word-level neural LMs for sentence completion.
This repository is built upon [Link](https://github.com/ctr4si/sentence-completion).

## Requirements
- numpy
- pandas
- tqdm
- pytorch == 1.1.0
- pytorch-transformers == 1.0.0
- sentencepiece (for tokenization of bert models)
- nltk == 3.3 (download `punkt` package for tokenization when experimenting WordRNNs)

## Datasets
- Microsoft Research Sentence Completion Challenge
    - Training and Test dataset can be downloaded from [Link](https://drive.google.com/open?id=0B5eGOMdyHn2mWDYtQzlQeGNKa2s). Store the downloaded test data in `data/completion/`.
- Scholastic Aptitude Test sentence completion questions
    - Collected questions are provided in [link](https://github.com/ctr4si/sentence-completion). Store the downloaded test data in `data/completion/`.
- TOPIK cloze questions
    - 10 samples are contained in `data/completion/topik_sample.csv`
    - Metadata for all questions are provided in `data/completion/topik_sample.csv`
    - You may request the full set via [e-mail](hee188@snu.ac.kr)
- Nineteenth century novels (19C novels)
    - A preprocessed dataset can be downloded from [link](https://github.com/ctr4si/sentence-completion).
- Sejong corpus can be downloaded through [link](https://ithub.korean.go.kr/user/total/database/corpusManager.do)

## Setup
- Pre-trained LM1B can be downloaded from [Link](https://github.com/tensorflow/models/tree/master/research/lm_1b)
- Pre-trained transformers of pytorch-transformers
    - automatically downloaded when running `eval_pretrained.py` with corresponding options 

create `./settings.json` containing
```json
{
  "prob_set_dir": "data/completion/",
  "prepro_dir": "path_to_prepro_dir",
  "lm1b_dir": "path_to_dir_containing_lm1b_model",
  "pretrans_dir": "path_to_dir_containing_pytorch_transformers",
  "sejong_dir": "path_to_dir_containing_sejong_corpus"
}
```

## Run
### Training of WordRNN
> python3 train.py --save_dir mynet

### Evaluation
- WordRNN
> python3 eval_trained.py --dir mynet

### Fine-tuning a Transformer-based model
> python3 finetune.py --model one_of_('bert', 'gpt', 'gpt2') --pretrained saved_name --update-embeddings

---
#### Acknowledgment
Thanks to [Sukhyun Cho](chosh90@snu.ac.kr) who manually collected and annotated the TOPIK questions
