import argparse
import os
import sys

from data.data_utils import (
    collate_finetune,
    CNNDailySeq2SeqDataset,
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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name_or_path)
    tok = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    collate_fn_test = partial(
        collate_finetune, pad_token_id=tok.pad_token_id, is_test=True
    )

    test_set = CNNDailySeq2SeqDataset(
        args.pretrained_model_name,
        split="test",
        summary_max_len=args.summary_truncate,
        article_max_len=args.article_truncate,
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
        model=model,
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
        "--pretrained_model_name_or_path", type=str, default="facebook/bart-large-cnn"
    )
    parser.add_argument(
        "--pretrained_model_type", type=str, default="bert"
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
