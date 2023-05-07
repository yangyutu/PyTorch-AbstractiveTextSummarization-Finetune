import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from module.finetune_model import FinetuneCausalLM
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import RobertaModel, RobertaTokenizer
from data.data_utils import (
    to_cuda,
    collate_mp_clm_finetune,
    CNNDailyCausalLMDataset,
)
from torch.utils.data import DataLoader
from functools import partial
from transformers import (
    BartTokenizer,
    AutoModel,
    AutoTokenizer,
    BartForConditionalGeneration,
    GPT2LMHeadModel,
)
from rouge import Rouge
import tqdm
from transformers import SummarizationPipeline, pipeline
import torch
import evaluate


tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2")

text1 = "The first prompt from the users"
text2 = "The second prompt"

texts = [text1, text2]

tokenizer.padding_side = "left"

# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# if the max_length is smaller than the actual input id sequence,
# the input ids will be truncated from the right to the left
encodings = tokenizer(
    texts, padding=True, max_length=3, truncation=True, return_tensors="pt"
)
prompt_text = "\nTL;DR:\n"
prompt_encoding = tokenizer(prompt_text, return_tensors="pt")
batch_size = len(texts)
dim = prompt_encoding["input_ids"].size(1)
encodings["input_ids"] = torch.cat(
    [encodings["input_ids"], prompt_encoding["input_ids"].expand(batch_size, dim)],
    axis=1,
)
encodings["attention_mask"] = torch.cat(
    [
        encodings["attention_mask"],
        prompt_encoding["attention_mask"].expand(batch_size, dim),
    ],
    axis=1,
)
print(encodings)
# parameters for generate method
# https://huggingface.co/docs/transformers/main_classes/text_generation
generated_ids = model.generate(
    inputs=encodings["input_ids"],
    attention_mask=encodings["attention_mask"],
    do_sample=True,
    top_k=10,
    top_p=0.5,
    temperature=0.8,
    max_new_tokens=10,  # this is the total sequence sequence, including the existing prompt length.
)
print(generated_ids)
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)
