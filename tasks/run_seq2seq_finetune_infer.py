import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from module.finetune_model import FinetuneSeq2SeqModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import RobertaModel, RobertaTokenizer
from data.data_utils import (
    collate_finetune,
    CNNDailySeq2SeqDataModule,
)
from torch.utils.data import DataLoader
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from rouge import Rouge
import tqdm
from transformers import pipeline
import torch
import evaluate


def main(args):
    # fix random seeds for reproducibility
    if args.baseline_no_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
    else:
        model = FinetuneSeq2SeqModel.load_from_checkpoint(args.model_ckpt).model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    data_module = CNNDailySeq2SeqDataModule(
        args.pretrained_model_name,
        summary_max_len=args.summary_truncate,
        article_max_len=args.article_truncate,
        add_prefix = args.add_prefix,
        add_suffix = args.add_suffix,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    
    test_dataloader =data_module.test_dataloader()

    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        batch_size=args.batch_size,
    )

    references = []
    final_outputs = []
    for i, batch in tqdm.tqdm(enumerate(test_dataloader)):
        articles = [item for item in batch["raw_input_text"]]
        outputs = summarizer(articles, max_length=130, min_length=30, do_sample=False)
        references.extend([item for item in batch["raw_target_text"]])
        for output in outputs:
            final_outputs.append(output["summary_text"].strip())
    #    break
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
    parser.add_argument("--summary_truncate", type=int, default=128)
    parser.add_argument("--article_truncate", type=int, default=512)
    
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument(
        "--pretrained_model_name", type=str, default="facebook/bart-large-cnn"
    )
    parser.add_argument("--add_prefix", type=str, default="")
    parser.add_argument("--add_suffix", type=str, default="")
    parser.add_argument("--model_ckpt", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--baseline_no_finetune", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
