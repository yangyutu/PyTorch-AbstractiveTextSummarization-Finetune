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
    collate_finetune,
    CNNDailySeq2SeqDataset,
)
from torch.utils.data import DataLoader
from functools import partial
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    AutoTokenizer,
)


def _load_data(args):

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    collate_fn = partial(
        collate_finetune, pad_token_id=tokenizer.pad_token_id, is_test=False
    )
    collate_fn_val = partial(
        collate_finetune, pad_token_id=tokenizer.pad_token_id, is_test=True
    )

    train_set = CNNDailySeq2SeqDataset(
        args.pretrained_model_name,
        split="train",
        summary_max_len=args.truncate,
        article_max_len=512,
        model_type=args.pretrained_model_type,
    )
    val_set = CNNDailySeq2SeqDataset(
        args.pretrained_model_name,
        split="validation",
        is_test=True,
        summary_max_len=512,
        article_max_len=512,
        model_type=args.pretrained_model_type,
    )

    print("finish dataset loading")
    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_val,
    )
    return train_dataloader, val_dataloader, tokenizer


def main(args):
    # fix random seeds for reproducibility
    SEED = args.seed
    pl.seed_everything(SEED)

    train_dataloader, val_dataloader, tokenizer = _load_data(args)
    config = {}
    config["lr"] = args.lr
    config["lr_warm_up_steps"] = args.lr_warm_up_steps

    if args.pretrained_model_type.lower() == "bart":
        pretrained_model = BartForConditionalGeneration.from_pretrained(
            args.pretrained_model_name, cache_dir="./local_cache"
        )
    elif args.pretrained_model_type.lower() == "t5":
        pretrained_model = T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model_name, cache_dir="./local_cache"
        )
    else:
        raise ValueError(f"{args.pretrained_model_type} is not supported")

    model = FinetuneSeq2SeqModel(
        config=config,
        pretrained_model=pretrained_model,
        pad_token_id=tokenizer.pad_token_id,
        label_smooth=0,
    )

    tags = [args.pretrained_model_name]
    if args.exp_tag:
        tags.append(args.exp_tag)
    wandb_logger = WandbLogger(
        project=args.project_name,  #
        log_model="all",
        save_dir=args.default_root_dir,
        tags=tags,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        precision=args.precision,
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.grad_accum,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            checkpoint_callback,
            lr_monitor,
        ],
        gradient_clip_val=args.max_grad_norm,
        deterministic=True,  # RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.
    )

    # trainer.validate(model, dataloaders=val_dataloader)
    # 4. Train!
    trainer.fit(model, train_dataloader, val_dataloader)


def parse_arguments():

    parser = argparse.ArgumentParser()

    # trainer specific arguments

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--default_root_dir", type=str, required=True)
    parser.add_argument("--exp_tag", type=str, default="")

    # # model specific arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_warm_up_steps", type=int, default=10000)
    parser.add_argument("--truncate", type=int, default=512)
    parser.add_argument("--pretrained_model_type", type=str, default="roberta-base")

    parser.add_argument("--pretrained_model_name", type=str, default="roberta-base")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
