import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from typing import Dict
import pytorch_lightning as pl
from model.losses import RankingLoss
from model.modeling_bart import BartScorer


class FinetuneSeq2SeqModel(pl.LightningModule):
    def __init__(self, config: Dict, pretrained_model_name, pad_token_id):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.pad_token_id = pad_token_id
        self.config = config

        self.mle_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, encoded_input, encoded_target):
        output = self.model(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            decoder_input_ids=encoded_target["input_ids"],
            decoder_attention_mask=encoded_target["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

        logits = output["logits"]  # [bz x cand_num, seq_len, word_dim]
        return logits

    def training_step(self, batch, batch_idx=0):
        logits_raw = self.forward(batch["input_text"], batch["target_text"])
        logits = logits_raw[:, :-1].reshape(
            -1, logits_raw.size(-1)
        )  # truncate last token
        gold = batch["target_text"]["input_ids"][:, 1:].reshape(-1)  # shift right
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
            batch["input_text"],
            batch["target_text"],
        )
        logits = logits_raw[:, :-1].reshape(
            -1, logits_raw.size(-1)
        )  # truncate last token
        gold = batch["target_text"]["input_ids"][:, 1:].reshape(-1)  # shift right
        loss = self.mle_fn(logits, gold)

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


class FinetuneWithContrastSeq2SeqModel(pl.LightningModule):
    def __init__(self, config: Dict, pretrained_model, pad_token_id, label_smooth=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = pretrained_model
        self.pad_token_id = pad_token_id
        self.config = config
        self.adding = self.config.get("adding", 0.0)
        self.length_penalty = self.config.get("length_penalty", 2.0)
        self.mle_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.ranking_loss = RankingLoss

        self.margin, self.gold_margin, self.gold_weight = 0.0, 0.0, 0.001
        self.ctr_loss_weight, self.mle_weight = 1.0, 1.0

    def forward(self, text_id, target_id):
        batch_size = text_id.size(0)

        new_text_id = []
        for _ in range(target_id.size(0)):
            new_text_id.append(text_id.expand(target_id.size(1), -1))

        text_id = torch.concat(new_text_id)
        target_id = target_id.reshape(-1, target_id.size(-1))

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

    def _compute_loss(self, batch):
        logits_raw = self.forward(batch["src_input_ids"], batch["candidate_ids"])
        batch_size = batch["src_input_ids"].size(0)
        logits_raw = logits_raw.view(
            batch_size, -1, logits_raw.size(1), logits_raw.size(2)
        )  # [bz, cand_num, seq_len, word_dim] # 17 cand, 1 is gold, other are the generated candidates

        logits_raw = logits_raw[
            :, :, :-1
        ]  # truncate last token # 1 x cand x (seq_len - 1) x word_dim
        gold_logits = logits_raw[
            :, 0
        ]  # logits_raw for the gold # 1 x seq_len x word_dim
        candidate_id = batch["candidate_ids"][:, :, 1:]  # shift right
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)

        logits_raw_normalized = F.log_softmax(logits_raw, dim=3)
        logits_at_candidate_tokens = torch.gather(
            logits_raw_normalized, 3, candidate_id
        ).squeeze(
            -1
        )  # [bz, cand_num, seq_len]

        cand_mask = cand_mask.float()
        scores = torch.mul(logits_at_candidate_tokens, cand_mask).sum(-1) / (
            (cand_mask.sum(-1) + self.adding) ** self.length_penalty
        )  # [bz, cand_num] ## score here is the sum of log prob of tokens in different steps

        candiate_score = scores[:, 1:]
        gold_score = scores[:, 0]

        gold_ids = batch["candidate_ids"][
            :, 0, 1:
        ]  # shift right # the first one is the gold reference

        mle_loss = self.mle_fn(
            gold_logits.view(-1, gold_logits.size(-1)), gold_ids.view(-1)
        )  # self.mle_fn(gold_logits, gold_ids)

        ctr_loss = self.ranking_loss(
            candiate_score, gold_score, self.margin, self.gold_margin, self.gold_weight
        )

        loss = self.ctr_loss_weight * ctr_loss + self.mle_weight * mle_loss

        return loss, mle_loss, ctr_loss, batch_size

    def training_step(self, batch, batch_idx=0):
        loss, mle_loss, ctr_loss, batch_size = self._compute_loss(batch)
        self.log_dict(
            {"ctr_loss": ctr_loss, "mle_loss": mle_loss, "loss": loss},
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx=0):
        loss, mle_loss, ctr_loss, batch_size = self._compute_loss(batch)
        self.log_dict(
            {"ctr_loss": ctr_loss, "mle_loss": mle_loss, "val_loss": loss},
            batch_size=batch_size,
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


class FinetuneWithContrastEfficientSeq2SeqModel(pl.LightningModule):
    def __init__(self, config: Dict, pretrained_model, pad_token_id, label_smooth=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = pretrained_model
        self.pad_token_id = pad_token_id
        self.config = config
        self.adding = self.config.get("adding", 0.0)
        self.length_penalty = self.config.get("length_penalty", 2.0)
        self.mle_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.ranking_loss = RankingLoss

        self.margin, self.gold_margin, self.gold_weight = 0.0, 0.0, 0.001
        self.ctr_loss_weight, self.mle_weight = 1.0, 1.0

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

    def _compute_loss(self, batch):
        logits_raw = self.forward(batch["src_input_ids"], batch["candidate_ids"])
        batch_size = batch["src_input_ids"].size(0)
        logits_raw = logits_raw.view(
            batch_size, -1, logits_raw.size(1), logits_raw.size(2)
        )  # [bz, cand_num, seq_len, word_dim] # 17 cand, 1 is gold, other are the generated candidates

        logits_raw = logits_raw[
            :, :, :-1
        ]  # truncate last token # 1 x cand x (seq_len - 1) x word_dim
        gold_logits = logits_raw[
            :, 0
        ]  # logits_raw for the gold # 1 x seq_len x word_dim
        candidate_id = batch["candidate_ids"][:, :, 1:]  # shift right
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)

        logits_raw_normalized = F.log_softmax(logits_raw, dim=3)
        logits_at_candidate_tokens = torch.gather(
            logits_raw_normalized, 3, candidate_id
        ).squeeze(
            -1
        )  # [bz, cand_num, seq_len]

        cand_mask = cand_mask.float()
        scores = torch.mul(logits_at_candidate_tokens, cand_mask).sum(-1) / (
            (cand_mask.sum(-1) + self.adding) ** self.length_penalty
        )  # [bz, cand_num] ## score here is the sum of log prob of tokens in different steps

        candiate_score = scores[:, 1:]
        gold_score = scores[:, 0]

        gold_ids = batch["candidate_ids"][
            :, 0, 1:
        ]  # shift right # the first one is the gold reference

        mle_loss = self.mle_fn(
            gold_logits.view(-1, gold_logits.size(-1)), gold_ids.view(-1)
        )  # self.mle_fn(gold_logits, gold_ids)

        ctr_loss = self.ranking_loss(
            candiate_score, gold_score, self.margin, self.gold_margin, self.gold_weight
        )

        loss = self.ctr_loss_weight * ctr_loss + self.mle_weight * mle_loss

        return loss, mle_loss, ctr_loss, batch_size

    def training_step(self, batch, batch_idx=0):
        # self.model.scoring_mode()
        loss, mle_loss, ctr_loss, batch_size = self._compute_loss(batch)
        self.log_dict(
            {"ctr_loss": ctr_loss, "mle_loss": mle_loss, "loss": loss},
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx=0):
        # self.model.scoring_mode()
        loss, mle_loss, ctr_loss, batch_size = self._compute_loss(batch)
        self.log_dict(
            {"ctr_loss": ctr_loss, "mle_loss": mle_loss, "val_loss": loss},
            batch_size=batch_size,
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


class FinetuneCausalLM(pl.LightningModule):
    def __init__(self, config: Dict, pretrained_model, pad_token_id):
        super().__init__()
        self.save_hyperparameters()
        self.model = pretrained_model
        self.pad_token_id = pad_token_id
        self.config = config

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
    from data.data_utils import CNNDailySeq2SeqDataset, collate_finetune

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


def _test_seq2seq_with_candidates_model_finetune():
    from torch.utils.data import DataLoader
    from transformers import (
        BartForConditionalGeneration,
        AutoTokenizer,
    )
    from functools import partial
    from data.data_utils import (
        CNNDailyWithCandidatesSeq2SeqDataset,
        collate_finetune_with_candidates,
    )

    model_name = "facebook/bart-large-cnn"
    fdir = "/mnt/d/MLData/data/summarization/cnndm_bart/cnndm/diverse/test/"
    data_set = CNNDailyWithCandidatesSeq2SeqDataset(
        fdir=fdir,
        tokenizer_name=model_name,
        is_test=True,
        summary_max_len=128,
        article_max_len=512,
        max_cand_num=16,
    )

    tok = AutoTokenizer.from_pretrained(model_name)

    collate_fn = partial(
        collate_finetune_with_candidates,
        pad_token_id=tok.pad_token_id,
        is_test=True,
    )

    dataloader = DataLoader(
        data_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    tok = AutoTokenizer.from_pretrained(model_name)

    config = {}
    config["lr"] = 2e-3
    config["lr_warm_up_steps"] = 10000

    pretrained_model = BartForConditionalGeneration.from_pretrained(
        model_name, cache_dir="./local_cache"
    )
    model = FinetuneWithContrastSeq2SeqModel(
        config, pretrained_model, tok.pad_token_id, 0.0
    )

    count = 0
    for batch in dataloader:
        # print(batch)
        loss = model.training_step(batch)
        break


def _test_seq2seq_with_candidates_efficient_model_finetune():
    from torch.utils.data import DataLoader
    from transformers import (
        BartForConditionalGeneration,
        AutoTokenizer,
    )
    from functools import partial
    from data.data_utils import (
        CNNDailyWithCandidatesSeq2SeqDataset,
        collate_finetune_with_candidates,
    )

    model_name = "facebook/bart-large-cnn"
    fdir = "/mnt/d/MLData/data/summarization/cnndm_bart/cnndm/diverse/test/"
    data_set = CNNDailyWithCandidatesSeq2SeqDataset(
        fdir=fdir,
        tokenizer_name=model_name,
        is_test=True,
        summary_max_len=128,
        article_max_len=512,
        max_cand_num=16,
    )

    tok = AutoTokenizer.from_pretrained(model_name)

    collate_fn = partial(
        collate_finetune_with_candidates,
        pad_token_id=tok.pad_token_id,
        is_test=True,
    )

    dataloader = DataLoader(
        data_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    tok = AutoTokenizer.from_pretrained(model_name)

    config = {}
    config["lr"] = 2e-3
    config["lr_warm_up_steps"] = 10000

    pretrained_model = BartScorer.from_pretrained(model_name, cache_dir="./local_cache")
    model = FinetuneWithContrastEfficientSeq2SeqModel(
        config, pretrained_model, tok.pad_token_id, 0.0
    )

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
    from data.data_utils import CNNDailyCausalLMDataset, collate_clm_finetune

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
    # _test_seq2seq_model_finetune()
    # _test_clm_model_finetune()
    # _test_seq2seq_with_candidates_model_finetune()
    _test_seq2seq_with_candidates_efficient_model_finetune()
