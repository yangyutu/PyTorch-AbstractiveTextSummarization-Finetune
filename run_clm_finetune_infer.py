import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from finetune_model import FinetuneCausalLM
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import RobertaModel, RobertaTokenizer
from data_utils import (
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


def _prepare_model_input_for_summarization(texts, tokenizer, prompt_text):

    # if the max_length is smaller than the actual input id sequence,
    # the input ids will be truncated from the right to the left
    encodings = tokenizer(
        texts, padding=True, max_length=512, truncation=True, return_tensors="pt"
    )

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
    return encodings


def main(args):
    # fix random seeds for reproducibility
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    collate_fn_test = partial(
        collate_mp_clm_finetune, pad_token_id=tokenizer.eos_token_id, is_test=True
    )

    test_set = CNNDailyCausalLMDataset(
        args.pretrained_model_name,
        split="test",
        summary_max_len=args.truncate,
        total_len=768,
        is_test=True,
    )

    print("finish dataset loading")

    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_test,
    )
    config = {}
    config["lr"] = args.lr
    config["lr_warm_up_steps"] = args.lr_warm_up_steps

    device = 0 if torch.cuda.is_available() else -1

    if args.baseline_no_finetune:
        pretrained_model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_model_name, cache_dir="./local_cache"
        )
        model = FinetuneCausalLM(
            config=config,
            pretrained_model=pretrained_model,
            pad_token_id=tokenizer.eos_token_id,
            label_smooth=0,
        )
    else:
        model = FinetuneCausalLM.load_from_checkpoint(args.model_ckpt)

    gpt_model = model.model

    # summarizer.tokenizer.pad_token_id = tokenizer.eos_token_id
    references = []
    final_outputs = []

    ### GPT-2 batch decoding
    # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    gpt_model.config.pad_token_id = gpt_model.config.eos_token_id

    ### The following is the manual batch decoding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt_model.to(device)
    prompt_text = "\nTL;DR:\n"
    for i, batch in tqdm.tqdm(enumerate(test_dataloader)):
        articles = [item["article"] for item in batch["data"]]
        references.extend([item["highlights"] for item in batch["data"]])
        encodings = _prepare_model_input_for_summarization(
            articles, tokenizer, prompt_text
        )
        encodings = encodings.to(device)
        with torch.no_grad():
            generated_ids = gpt_model.generate(
                inputs=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                do_sample=True,
                top_k=10,
                top_p=0.5,
                temperature=0.8,
                max_new_tokens=130,
                min_new_tokens=30,
                no_repeat_ngram_size=3,
            )
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            for output in generated_texts:
                start_idx = output.find(prompt_text)
                final_outputs.append(output[start_idx + len(prompt_text) :])

    # print(articles)
    # print(final_outputs)

    # print(final_outputs)
    rouge_scorer = Rouge()
    score = rouge_scorer.get_scores(final_outputs, references, avg=True)
    print(score)

    rouge = evaluate.load("rouge")
    score = rouge.compute(predictions=final_outputs, references=references)
    print(score)


def parse_arguments():

    parser = argparse.ArgumentParser()

    # trainer specific arguments

    parser.add_argument("--seed", type=int, default=1)
    # parser.add_argument("--data_dir", type=str, required=True)
    # parser.add_argument("--dataset_name", type=str, required=True)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--exp_tag", type=str, default="")

    # # model specific arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_warm_up_steps", type=int, default=10000)
    parser.add_argument("--truncate", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument(
        "--pretrained_model_name", type=str, default="facebook/bart-large-cnn"
    )
    parser.add_argument(
        "--pretrained_model_type", type=str, default="facebook/bart-large-cnn"
    )
    parser.add_argument("--model_ckpt", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--baseline_no_finetune", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(args)
