import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from finetune_model import FinetuneSeq2SeqModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import RobertaModel, RobertaTokenizer
from data_utils import (
    to_cuda,
    collate_mp_finetune,
    FineTuneSeq2SeqDataset,
    CNNDailySeq2SeqDataset,
)
from torch.utils.data import DataLoader
from functools import partial
from transformers import (
    BartTokenizer,
    AutoModel,
    AutoTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
)
from rouge import Rouge
import tqdm
from transformers import SummarizationPipeline, pipeline
import torch
import evaluate


def baseline_no_finetune_using_pipeline_old(args):
    # fix random seeds for reproducibility
    tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_name)
    collate_fn_test = partial(
        collate_mp_finetune, pad_token_id=tokenizer.pad_token_id, is_test=True
    )

    test_set = FineTuneSeq2SeqDataset(
        f"{args.data_dir}/test",
        args.pretrained_model_name,
        is_test=True,
        max_len=512,
        total_len=512,
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
    pretrained_model = BartForConditionalGeneration.from_pretrained(
        args.pretrained_model_name, cache_dir="./local_cache"
    )
    model = FinetuneSeq2SeqModel(
        config=config,
        pretrained_model=pretrained_model,
        pad_token_id=tokenizer.pad_token_id,
        label_smooth=0,
    )
    summarizer = pipeline(
        "summarization",
        model=model.model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        batch_size=args.batch_size,
    )

    references = []
    final_outputs = []
    for i, batch in tqdm.tqdm(enumerate(test_dataloader)):
        articles = [" ".join(item["article"]) for item in batch["data"]]
        outputs = summarizer(articles, max_length=130, min_length=30, do_sample=False)
        references.extend([" ".join(item["abstract"]) for item in batch["data"]])
        for output in outputs:
            final_outputs.append(output["summary_text"].strip())
        break
    print(final_outputs)
    # rouge_scorer = Rouge()
    # score = rouge_scorer.get_scores(final_outputs, references, avg=True)
    rouge = evaluate.load("rouge")
    score = rouge.compute(predictions=final_outputs, references=references)
    print(score)


def baseline_no_finetune_using_pipeline(args):
    # fix random seeds for reproducibility
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    collate_fn_test = partial(
        collate_mp_finetune, pad_token_id=tokenizer.pad_token_id, is_test=True
    )

    test_set = CNNDailySeq2SeqDataset(
        args.pretrained_model_name,
        split="test",
        summary_max_len=args.truncate,
        article_max_len=512,
        is_test=True,
        model_type=args.pretrained_model_type,
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
    if args.pretrained_model_type.lower() == "bart":
        pretrained_model = BartForConditionalGeneration.from_pretrained(
            args.pretrained_model_name, cache_dir="./local_cache"
        )
    elif args.pretrained_model_type.lower() == "t5":
        pretrained_model = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_name, cache_dir="./local_cache"
        )
    model = FinetuneSeq2SeqModel(
        config=config,
        pretrained_model=pretrained_model,
        pad_token_id=tokenizer.pad_token_id,
        label_smooth=0,
    )
    summarizer = pipeline(
        "summarization",
        model=model.model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        batch_size=args.batch_size,
    )

    references = []
    final_outputs = []
    for i, batch in tqdm.tqdm(enumerate(test_dataloader)):
        articles = [item["article"] for item in batch["data"]]
        outputs = summarizer(articles, max_length=130, min_length=30, do_sample=False)
        references.extend([item["highlights"] for item in batch["data"]])
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


def main(args):
    # fix random seeds for reproducibility

    model = FinetuneSeq2SeqModel.load_from_checkpoint(args.model_ckpt)
    tok = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    collate_fn_test = partial(
        collate_mp_finetune, pad_token_id=tok.pad_token_id, is_test=True
    )

    test_set = CNNDailySeq2SeqDataset(
        args.pretrained_model_name,
        split="test",
        summary_max_len=args.truncate,
        article_max_len=512,
        is_test=True,
        model_type=args.pretrained_model_type,
    )

    print("finish dataset loading")

    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_test,
    )

    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model=model.model,
        tokenizer=tok,
        device=device,
        truncation=True,
        batch_size=args.batch_size,
    )

    references = []
    final_outputs = []
    for i, batch in tqdm.tqdm(enumerate(test_dataloader)):
        articles = [item["article"] for item in batch["data"]]
        outputs = summarizer(articles, max_length=130, min_length=30, do_sample=False)
        references.extend([item["highlights"] for item in batch["data"]])
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
    if args.baseline_no_finetune:
        # baseline_no_finetune(args)
        # baseline_no_finetune_using_pipeline(args)
        baseline_no_finetune_using_pipeline(args)
        # baseline_official(args)
    else:
        main(args)
