# -*- coding: utf-8 -*-
"""spanberta-pretraining-bert-from-scratch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mXWYYkB9UjRdklPVSDvAcUDralmv3Pgv

# SpanBERTa - Part I: How We Trained RoBERTa Language Model for Spanish from Scratch

# Introduction

- [Part II: Fine-tuning SpanBERTa for Named Entity Recognition](https://chriskhanhtran.github.io/posts/named-entity-recognition-with-transformers/)

Self-training methods with transformer models have achieved state-of-the-art performance on most NLP tasks. However, because training them is computationally expensive, most currently available pretrained transformer models are only for English. Therefore, to improve performance in NLP tasks in our projects on Spanish, my team at [Skim AI](https://skimai.com/) decided to train a **RoBERTa** language model for Spanish from scratch and call it SpanBERTa.

SpanBERTa has the same size as RoBERTa-base. We followed RoBERTa's training schema to train the model on 18 GB of [OSCAR](https://traces1.inria.fr/oscar/)'s Spanish corpus in 8 days using 4 Tesla P100 GPUs.

In this blog post, we will walk through an end-to-end process to train a BERT-like language model from scratch using `transformers` and `tokenizers` libraries by Hugging Face. There is also a Google Colab notebook to run the codes in this article directly. You can also modify the notebook accordingly to train a BERT-like model for other languages or fine-tune it on your customized dataset.

Before moving on, I want to express a huge thank to the Hugging Face team for making state-of-the-art NLP models accessible for everyone.

# Setup

## 1. Install Dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip uninstall -y tensorflow
# !pip install transformers==2.8.0

"""## 2. Data

We pretrained SpanBERTa on [OSCAR](https://traces1.inria.fr/oscar/)'s Spanish corpus. The full size of the dataset is 150 GB and we used a portion of 18 GB to train.

In this example, for simplicity, we will use a dataset of Spanish movie subtitles from [OpenSubtitles](https://www.opensubtitles.org/en/search). This dataset has a size of 5.4 GB and we will train on a subset of ~300 MB.
"""

import os

# Download and unzip movie substitle dataset
if not os.path.exists('data/dataset.txt'):
  !wget "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/mono/es.txt.gz" -O dataset.txt.gz
  !gzip -d dataset.txt.gz
  !mkdir data
  !mv dataset.txt data

# Total number of lines and some random lines
!wc -l data/dataset.txt
!shuf -n 5 data/dataset.txt

# Get a subset of first 1,000,000 lines for training
TRAIN_SIZE = 1000000 #@param {type:"integer"}
!(head -n $TRAIN_SIZE data/dataset.txt) > data/train.txt

# Get a subset of next 10,000 lines for validation
VAL_SIZE = 10000 #@param {type:"integer"}
!(sed -n {TRAIN_SIZE + 1},{TRAIN_SIZE + VAL_SIZE}p data/dataset.txt) > data/dev.txt

"""## 3. Train a Tokenizer

The original BERT implementation uses a WordPiece tokenizer with a vocabulary of 32K subword units. This method, however, can introduce "unknown" tokens when processing rare words.

In this implementation, we use a byte-level BPE tokenizer with a vocabulary of 50,265 subword units (same as RoBERTa-base). Using byte-level BPE makes it possible to learn a subword vocabulary of modest size that can encode any input without getting "unknown" tokens.

Because `ByteLevelBPETokenizer` produces 2 files `["vocab.json", "merges.txt"]` while `BertWordPieceTokenizer` produces only 1 file `vocab.txt`, it will cause an error if we use `BertWordPieceTokenizer` to load outputs of a BPE tokenizer.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from tokenizers import ByteLevelBPETokenizer
# 
# path = "data/train.txt"
# 
# # Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()
# 
# # Customize training
# tokenizer.train(files=path,
#                 vocab_size=50265,
#                 min_frequency=2,
#                 special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
# 
# # Save files to disk
# !mkdir -p "models/roberta"
# tokenizer.save("models/roberta")

"""Super fast! It takes only 2 minutes to train on 10 million lines.

<img src="https://github.com/chriskhanhtran/spanish-bert/blob/master/img/train_tokenizers.gif?raw=true" width="700">

# Traing Language Model from Scratch

## 1. Model Architecture

RoBERTa has exactly the same architecture as BERT. The only differences are:
- RoBERTa uses a Byte-Level BPE tokenizer with a larger subword vocabulary (50k vs 32k).
- RoBERTa implements dynamic word masking and drops next sentence prediction task.
- RoBERTa's training hyperparameters.

Other architecture configurations can be found in the documentation ([RoBERTa](https://huggingface.co/transformers/_modules/transformers/configuration_roberta.html#RobertaConfig), [BERT](https://huggingface.co/transformers/_modules/transformers/configuration_bert.html#BertConfig)).
"""

import json
config = {
	"architectures": [
		"RobertaForMaskedLM"
	],
	"attention_probs_dropout_prob": 0.1,
	"hidden_act": "gelu",
	"hidden_dropout_prob": 0.1,
	"hidden_size": 768,
	"initializer_range": 0.02,
	"intermediate_size": 3072,
	"layer_norm_eps": 1e-05,
	"max_position_embeddings": 514,
	"model_type": "roberta",
	"num_attention_heads": 12,
	"num_hidden_layers": 12,
	"type_vocab_size": 1,
	"vocab_size": 50265
}

with open("models/roberta/config.json", 'w') as fp:
    json.dump(config, fp)

tokenizer_config = {"max_len": 512}

with open("models/roberta/tokenizer_config.json", 'w') as fp:
    json.dump(tokenizer_config, fp)

"""## 2. Training Hyperparameters

| Hyperparam          | BERT-base | RoBERTa-base |
|---------------------|:---------:|:------------:|
|Sequence Length      | 128, 512  | 512          |
|Batch Size           | 256       | 8K           |
|Peak Learning Rate   | 1e-4      | 6e-4         |
|Max Steps            | 1M        | 500K         |
|Warmup Steps         | 10K       | 24K          |
|Weight Decay         | 0.01      | 0.01         |
|Adam $\epsilon$      | 1e-6      | 1e-6         |
|Adam $\beta_1$       | 0.9       | 0.9          |
|Adam $\beta_2$       | 0.999     | 0.98         |
|Gradient Clipping    | 0.0       | 0.0          |

Note the batch size when training RoBERTa is 8000. Therefore, although RoBERTa-base was trained for 500K steps, its training computational cost is 16 times that of BERT-base. In the [RoBERTa paper](https://arxiv.org/pdf/1907.11692.pdf), it is shown that training with large batches improves perplexity for the masked language modeling objective, as well as end-task accuracy. Larger batch size can be obtained by tweaking `gradient_accumulation_steps`.

Due to computational constraint, we followed BERT-base's training schema and trained our SpanBERTa model using 4 Tesla P100 GPUs for 200K steps in 8 days.

## 3. Start Training

We will train our model from scratch using [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py), a script provided by Hugging Face, which will preprocess, tokenize the corpus and train the model on *Masked Language Modeling* task. The script is optimized to train on a single big corpus. Therefore, if your dataset is large and you want to split it to train sequentially, you will need to modify the script, or be ready to get a monster machine with high memory.
"""

# Update April 22, 2020: Hugging Face updated run_language_modeling.py script.
# Please use this version which was before the update.
!wget -c https://raw.githubusercontent.com/chriskhanhtran/spanish-bert/master/run_language_modeling.py

"""**Important Arguments**
- `--line_by_line` Whether distinct lines of text in the dataset are to be handled as distinct sequences. If each line in your dataset is long and has ~512 tokens or more, you should use this setting. If each line is short, the default text preprocessing will concatenate all lines, tokenize them and slit tokenized outputs into blocks of 512 tokens. You can also split your datasets into small chunks and preprocess them separately. 3GB of text will take ~50 minutes to process with the default `TextDataset` class.
- `--should_continue` Whether to continue from latest checkpoint in output_dir.
- `--model_name_or_path` The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.
- `--mlm` Train with masked-language modeling loss instead of language modeling.
- `--config_name, --tokenizer_name` Optional pretrained config and tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new config.
- `--per_gpu_train_batch_size` Batch size per GPU/CPU for training. Choose the largest number you can fit on your GPUs. You will see an error if your batch size is too large.
- `--gradient_accumulation_steps` Number of updates steps to accumulate before performing a backward/update pass. You can use this trick to increase batch size. For example, if `per_gpu_train_batch_size = 16` and `gradient_accumulation_steps = 4`, your total train batch size will be 64.
- `--overwrite_output_dir` Overwrite the content of the output directory.
- `--no_cuda, --fp16, --fp16_opt_level` Arguments for training on GPU/CPU.
- Other arguments are model paths and training hyperparameters.

It's highly recommended to include model type (eg. "roberta", "bert", "gpt2" etc.) in the model path because the script uses the [`AutoModels`](https://huggingface.co/transformers/model_doc/auto.html?highlight=automodels) class to guess the model's configuration using pattern matching on the provided path.
"""

# Model paths
MODEL_TYPE = "roberta" #@param ["roberta", "bert"]
MODEL_DIR = "models/roberta" #@param {type: "string"}
OUTPUT_DIR = "models/roberta/output" #@param {type: "string"}
TRAIN_PATH = "data/train.txt" #@param {type: "string"}
EVAL_PATH = "data/dev.txt" #@param {type: "string"}

"""For this example, we will train for only 25 steps on a Tesla P4 GPU provided by Colab."""

!nvidia-smi

# Command line
cmd = """python run_language_modeling.py \
    --output_dir {output_dir} \
    --model_type {model_type} \
    --mlm \
    --config_name {config_name} \
    --tokenizer_name {tokenizer_name} \
    {line_by_line} \
    {should_continue} \
    {model_name_or_path} \
    --train_data_file {train_path} \
    --eval_data_file {eval_path} \
    --do_train \
    {do_eval} \
    {evaluate_during_training} \
    --overwrite_output_dir \
    --block_size 512 \
    --max_step 25 \
    --warmup_steps 10 \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 100.0 \
    --save_total_limit 10 \
    --save_steps 10 \
    --logging_steps 2 \
    --seed 42
"""

# Arguments for training from scratch. I turn off evaluate_during_training,
#   line_by_line, should_continue, and model_name_or_path.
train_params = {
    "output_dir": OUTPUT_DIR,
    "model_type": MODEL_TYPE,
    "config_name": MODEL_DIR,
    "tokenizer_name": MODEL_DIR,
    "train_path": TRAIN_PATH,
    "eval_path": EVAL_PATH,
    "do_eval": "--do_eval",
    "evaluate_during_training": "",
    "line_by_line": "",
    "should_continue": "",
    "model_name_or_path": "",
}

"""If you are training on a virtual machine, you can install tensorboard to monitor the training process. Here is our [Tensorboard](https://tensorboard.dev/experiment/4wOFJBwPRBK9wjKE6F32qQ/#scalars) for training SpanBERTa.

```sh
pip install tensorboard==2.1.0
tensorboard dev upload --logdir runs
```

<img src="https://github.com/chriskhanhtran/spanish-bert/blob/master/img/tensorboard-spanberta.JPG?raw=true" width="400">

*After 200k steps, the loss reached 1.8 and the perplexity reached 5.2.*

Now let's start training!
"""

!{cmd.format(**train_params)}

"""## 4. Predict Masked Words

After training your language model, you can upload and share your model with the community. We have uploaded our SpanBERTa model to Hugging Face's server. Before evaluating the model on downstream tasks, let's see how it has learned to fill masked words given a context.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# %%time
# from transformers import pipeline
# 
# fill_mask = pipeline(
#     "fill-mask",
#     model="chriskhanhtran/spanberta",
#     tokenizer="chriskhanhtran/spanberta"
# )

"""I pick a sentence from Wikipedia's article about COVID-19.

The original sentence is "*Lavarse frecuentemente las manos con agua y jab??n,*" meaning "*Frequently wash your hands with soap and water.*"

The masked word is **"jab??n" (soap)** and the top 5 predictions are **soap, salt, steam, lemon** and **vinegar**. It is interesting that the model somehow learns that we should wash our hands with things that can kill bacteria or contain acid.
"""

fill_mask("Lavarse frecuentemente las manos con agua y <mask>.")

"""# Conclusion

We have walked through how to train a BERT language model for Spanish from scratch and seen that the model has learned properties of the language by trying to predict masked words given a context. You can also follow this article to fine-tune a pretrained BERT-like model on your customized dataset.

Next, we will implement the pretrained models on downstream tasks including Sequence Classification, NER, POS tagging, and NLI, as well as compare the model's performance with some non-BERT models.

Stay tuned for our next posts!
"""