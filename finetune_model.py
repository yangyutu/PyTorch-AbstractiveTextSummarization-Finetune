import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
from label_smoothing_loss import label_smoothing_loss
from typing import Dict
import pytorch_lightning as pl


class FinetuneSeq2SeqModel(pl.LightningModule):
    def __init__(self, config: Dict, pretrained_model, pad_token_id, label_smooth=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = pretrained_model
        self.pad_token_id = pad_token_id
        self.config = config

        if label_smooth > 0:
            self.mle_fn = label_smoothing_loss(
                ignore_index=pad_token_id, epsilon=label_smooth
            )
        else:
            self.mle_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, text_id, target_id):

        batch_size = text_id.size(0)

        input_mask = text_id != self.pad_token_id
        target_mask = target_id != self.pad_token_id
        # cand_mask[:, :, 0] = 1
        output = self.model(
            input_ids=text_id,
            attention_mask=input_mask,
            decoder_input_ids=target_id,
            decoder_attention_mask=target_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = output["logits"]  # [bz x cand_num, seq_len, word_dim]
        return logits

    def training_step(self, batch, batch_idx=0):

        logits_raw = self.forward(batch["src_input_ids"], batch["target_ids"])
        logits = logits_raw[:, :-1].reshape(
            -1, logits_raw.size(-1)
        )  # truncate last token
        gold = batch["target_ids"][:, 1:].reshape(-1)  # shift right
        loss = self.mle_fn(logits, gold)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx=0):

        logits_raw = self.forward(
            batch["src_input_ids"],
            batch["target_ids"],
        )
        logits = logits_raw[:, :-1].reshape(
            -1, logits_raw.size(-1)
        )  # truncate last token
        gold = batch["target_ids"][:, 1:].reshape(-1)  # shift right
        loss = self.mle_fn(logits, gold)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def generate(
        self,
        src_input_ids,
        min_length,
        max_length,
        do_sample=False,
        **kwargs,
    ):

        input_mask = src_input_ids != self.pad_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=src_input_ids,
                attention_mask=input_mask,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                **kwargs,
            )
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config["lr"])
        lr_warmup_steps = self.config["lr_warm_up_steps"]

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class FinetuneCausalLM(pl.LightningModule):
    def __init__(self, config: Dict, pretrained_model, pad_token_id, label_smooth=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = pretrained_model
        self.pad_token_id = pad_token_id
        self.config = config

        if label_smooth > 0:
            self.mle_fn = label_smoothing_loss(
                ignore_index=pad_token_id, epsilon=label_smooth
            )
        else:
            self.mle_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, batch):
        inputs = batch["input_ids"]
        start_idx = batch["start_idx"]
        # 1 for real tokens and 0 for padded tokens
        mask = (inputs != self.pad_token_id).float()
        output = self.model(
            input_ids=inputs,
            attention_mask=mask,
            return_dict=True,
        )

        logits = output["logits"]  # [bz x cand_num, seq_len, word_dim]
        shift_logits = torch.cat(
            [logits[i, start_idx[i] : -1, :] for i in range(logits.size(0))], axis=0
        )
        # shift right as the target
        shift_labels = torch.cat(
            [inputs[i, start_idx[i] + 1 :] for i in range(inputs.size(0))], axis=0
        )

        loss = self.mle_fn(shift_logits, shift_labels)

        return loss

    def training_step(self, batch, batch_idx=0):

        loss = self.forward(batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx=0):

        loss = self.forward(batch)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config["lr"])
        lr_warmup_steps = self.config["lr_warm_up_steps"]

        def lr_foo(step):
            lr_scale = min((step + 1) ** (-0.5), step / lr_warmup_steps ** (1.5))

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _test_seq2seq_model_finetune():
    from torch.utils.data import DataLoader
    from transformers import (
        T5ForConditionalGeneration,
        AutoTokenizer,
    )
    from functools import partial
    from data_utils import CNNDailySeq2SeqDataset, collate_finetune

    pretrained_model_name = "google/flan-t5-base"
    data_set = CNNDailySeq2SeqDataset(
        pretrained_model_name,
        is_test=False,
        summary_max_len=512,
        article_max_len=1024,
        model_type="t5",
    )
    print(data_set[0])

    tok = AutoTokenizer.from_pretrained(pretrained_model_name)

    collate_fn = partial(
        collate_finetune,
        pad_token_id=tok.pad_token_id,
        is_test=False,
    )

    dataloader = DataLoader(
        data_set, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    config = {}
    config["lr"] = 2e-3
    config["lr_warm_up_steps"] = 10000

    pretrained_model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name, cache_dir="./local_cache"
    )
    model = FinetuneSeq2SeqModel(config, pretrained_model, tok.pad_token_id, 0.0)

    count = 0
    for batch in dataloader:
        # print(batch)
        loss = model.training_step(batch)
        break


def _test_clm_model_finetune():
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        GPT2LMHeadModel,
        AutoTokenizer,
    )
    from functools import partial
    from data_utils import CNNDailyCausalLMDataset, collate_clm_finetune

    pretrained_model_name = "gpt2"
    data_set = CNNDailyCausalLMDataset(
        pretrained_model_name,
        is_test=False,
        summary_max_len=128,
        total_len=768,
    )
    print(data_set[0])

    tok = AutoTokenizer.from_pretrained(pretrained_model_name)

    collate_fn = partial(
        collate_clm_finetune,
        is_test=False,
        pad_token_id=tok.eos_token_id,
    )

    dataloader = DataLoader(
        data_set, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    config = {}
    config["lr"] = 2e-3
    config["lr_warm_up_steps"] = 10000

    pretrained_model = GPT2LMHeadModel.from_pretrained(
        pretrained_model_name, cache_dir="./local_cache"
    )
    model = FinetuneCausalLM(
        config, pretrained_model, pad_token_id=tok.eos_token_id, label_smooth=0.0
    )

    count = 0
    for batch in dataloader:
        # print(batch)
        loss = model.training_step(batch)
        break


if __name__ == "__main__":
    # _test_BART_finetune()
    _test_seq2seq_model_finetune()
    _test_clm_model_finetune()
