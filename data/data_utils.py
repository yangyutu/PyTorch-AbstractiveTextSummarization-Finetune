from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer, AutoTokenizer
import pytorch_lightning as pl
from functools import partial
from datasets import load_dataset
import os

# because we are performing tokenization inside each worker, we need to disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


def collate_finetune(batch, pad_token_id, is_test=False):
    # This collate function is mainly used for padding
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for i, x in enumerate(X):
            result[i, : x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    target_ids = [x["target_ids"] for x in batch]
    max_len = max([len(x) for x in target_ids])
    target_ids = pad(target_ids, max_len)
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "target_ids": target_ids,
    }
    if is_test:
        result["data"] = data
    return result


def pair_text_collate_with_tokenization(
    data,
    tokenizer,
    encoder_truncate,
    decoder_truncate,
    is_test=False,
):
    input_texts, target_texts = zip(*data)
    encoded_input = tokenizer(
        list(input_texts),
        return_tensors="pt",
        max_length=encoder_truncate,
        truncation=True,
        padding=True,
    )
    encoded_target = tokenizer(
        list(target_texts),
        return_tensors="pt",
        max_length=decoder_truncate,
        truncation=True,
        padding=True,
    )

    if not is_test:
        return {"input_text": encoded_input, "target_text": encoded_target}
    else:
        return {
            "input_text": encoded_input,
            "target_text": encoded_target,
            "raw_input_text": input_texts,
            "raw_target_text": target_texts,
        }


class CNNDailyPairTextDataset(Dataset):
    def __init__(self, split="train", add_prefix="", add_suffix=""):
        self.dataset = load_dataset("cnn_dailymail", version="3.0.0")[split]
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_text = self.dataset[idx]["article"]
        target_text = self.dataset[idx]["highlights"].replace("\n", " ")
        if self.add_prefix:
            input_text = self.add_prefix + " " + input_text
        if self.add_suffix:
            target_text = input_text + " " + self.add_suffix
        return input_text, target_text


class CNNDailySeq2SeqDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name,
        article_max_len=512,
        summary_max_len=128,
        add_prefix="",
        add_suffix="",
        batch_size=128,
        num_workers=16,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.article_max_len = article_max_len
        self.summary_max_len = summary_max_len
        self.add_prefix = add_prefix
        self.add_suffix = add_suffix
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = CNNDailyPairTextDataset(
            split="train", add_prefix=self.add_prefix, add_suffix=self.add_suffix
        )
        collate_fn = partial(
            pair_text_collate_with_tokenization,
            tokenizer=self.tokenizer,
            encoder_truncate=self.article_max_len,
            decoder_truncate=self.summary_max_len,
            is_test=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        dataset = CNNDailyPairTextDataset(
            split="validation", add_prefix=self.add_prefix, add_suffix=self.add_suffix
        )
        collate_fn = partial(
            pair_text_collate_with_tokenization,
            tokenizer=self.tokenizer,
            encoder_truncate=self.article_max_len,
            decoder_truncate=self.summary_max_len,
            is_test=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        dataset = CNNDailyPairTextDataset(
            split="test", add_prefix=self.add_prefix, add_suffix=self.add_suffix
        )
        collate_fn = partial(
            pair_text_collate_with_tokenization,
            tokenizer=self.tokenizer,
            encoder_truncate=self.article_max_len,
            decoder_truncate=self.summary_max_len,
            is_test=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )


class CNNDailyCausalLMDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split="train",
        summary_max_len=128,
        is_test=False,
        total_len=768,
        add_suffix="\nTL;DR:\n",
    ):
        self.dataset = load_dataset("cnn_dailymail", version="3.0.0")[split]
        self.tokenizer = tokenizer
        self.summary_max_len = summary_max_len
        self.is_test = is_test
        self.total_len = total_len
        self.add_suffix = add_suffix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_txt = self.dataset[idx]["article"]

        prompt_ids = self.tokenizer.encode(self.add_suffix, add_special_tokens=False)

        src_input_ids = self.tokenizer.encode(
            src_txt, add_special_tokens=False, truncation=True, max_length=512
        )
        abstract_txt = self.dataset[idx]["highlights"].replace("\n", " ")

        abstract_input_ids = self.tokenizer.encode(
            abstract_txt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.summary_max_len,
        )

        src_allow_len = min(
            self.total_len - len(abstract_input_ids) - len(prompt_ids),
            len(src_input_ids),
        )

        input_ids = src_input_ids[:src_allow_len] + prompt_ids + abstract_input_ids

        result = {
            "input_ids": torch.LongTensor(input_ids),
            #"start_idx": src_allow_len + len(prompt_ids),
            "start_idx": self.total_len - len(abstract_input_ids),
        }
        if self.is_test:
            result["data"] = {
                "article": src_txt,
                "highlights": abstract_txt,
            }
        return result


class CNNDailyCausalLMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name,
        summary_max_len=512,
        total_len=768,
        add_suffix="",
        batch_size=128,
        num_workers=16,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.summary_max_len = summary_max_len
        self.total_len = total_len
        self.add_suffix = add_suffix
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = CNNDailyCausalLMDataset(
            tokenizer=self.tokenizer,
            split="train",
            add_suffix=self.add_suffix,
            summary_max_len=self.summary_max_len,
            total_len=self.total_len,
        )
        collate_fn = partial(
            collate_clm_finetune,
            pad_token_id=self.tokenizer.pad_token_id,
            is_test=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        dataset = CNNDailyCausalLMDataset(
            tokenizer=self.tokenizer,
            split="validation",
            add_suffix=self.add_suffix,
            summary_max_len=self.summary_max_len,
            total_len=self.total_len,
        )
        collate_fn = partial(
            collate_clm_finetune,
            pad_token_id=self.tokenizer.pad_token_id,
            is_test=False,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        dataset = CNNDailyCausalLMDataset(
            tokenizer=self.tokenizer,
            split="test",
            add_suffix=self.add_suffix,
            summary_max_len=self.summary_max_len,
            total_len=self.total_len,
        )
        collate_fn = partial(
            collate_clm_finetune,
            pad_token_id=self.tokenizer.pad_token_id,
            is_test=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )


def collate_finetune_with_candidates(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for i, x in enumerate(X):
            result[i, : x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    candidate_ids = torch.stack(candidate_ids)
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
    }
    if is_test:
        result["data"] = data
    return result


def collate_clm_finetune(batch, pad_token_id=-100, is_test=False):
    # This collate function is mainly used for padding
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        # left padding
        for i, x in enumerate(X):
            result[i, -x.size(0):] = x
        return result

    input_ids = pad([x["input_ids"] for x in batch])
    start_idx = [x["start_idx"] for x in batch]
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "input_ids": input_ids,
        "start_idx": start_idx,
    }
    if is_test:
        result["data"] = data
    return result


class CNNDailyWithCandidatesSeq2SeqDataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer_name,
        is_test=False,
        summary_max_len=-1,
        article_max_len=512,
        is_sorted=True,
        max_cand_num=-1,
        is_untok=True,
        num=-1,
    ):
        """data format: article, abstract, [(candidiate_i, score_i)]"""
        self.isdir = os.path.isdir(data_dir)
        if self.isdir:
            self.data_dir = data_dir
            if num > 0:
                self.num = min(len(os.listdir(data_dir)), num)
            else:
                self.num = len(os.listdir(data_dir))
        else:
            with open(data_dir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        self.tok = AutoTokenizer.from_pretrained(
            tokenizer_name, verbose=False, model_max_length=512
        )
        self.summary_max_len = summary_max_len
        self.is_test = is_test
        self.article_max_len = article_max_len
        self.sorted = is_sorted
        self.max_cand_num = max_cand_num
        self.is_untok = is_untok

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.data_dir, "%d.json" % idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        if self.is_untok:
            article = data["article_untok"]
        else:
            article = data["article"]
        src_txt = " ".join(article)
        src = self.tok.batch_encode_plus(
            [src_txt],
            max_length=self.article_max_len,
            return_tensors="pt",
            pad_to_max_length=False,
            truncation=True,
        )
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
        if self.is_untok:
            abstract = data["abstract_untok"]
        else:
            abstract = data["abstract"]
        if self.max_cand_num > 0:
            candidates = data["candidates_untok"][: self.max_cand_num]
            _candidates = data["candidates"][: self.max_cand_num]
            data["candidates"] = _candidates
        if self.sorted:
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            _candidates = sorted(_candidates, key=lambda x: x[1], reverse=True)
            data["candidates"] = _candidates
        if not self.is_untok:
            candidates = _candidates

        # cand_txt consists of the gold reference plus other candidates
        cand_txt = [" ".join(abstract)] + [" ".join(x[0]) for x in candidates]
        cand = self.tok.batch_encode_plus(
            cand_txt,
            max_length=self.summary_max_len,
            return_tensors="pt",
            pad_to_max_length=False,
            truncation=True,
            padding=True,
        )
        candidate_ids = cand["input_ids"]

        result = {
            "src_input_ids": src_input_ids,
            "candidate_ids": candidate_ids,
        }
        if self.is_test:
            result["data"] = data
        return result


def _test_finetune_seq2seq_data():
    dataloader = CNNDailySeq2SeqDataModule(
        tokenizer_name="t5-large", article_max_len=768, summary_max_len=128
    ).train_dataloader()

    for batch in dataloader:
        print(batch)
        break


def _test_finetune_seq2seq_with_candidates_data():
    model_name = "facebook/bart-large-cnn"
    data_dir = "/mnt/d/MLData/data/summarization/cnndm_bart/cnndm/diverse/test/"
    data_set = CNNDailyWithCandidatesSeq2SeqDataset(
        data_dir=data_dir,
        tokenizer_name=model_name,
        is_test=True,
        summary_max_len=128,
        article_max_len=512,
        max_cand_num=16,
    )

    print(data_set[0])
    tok = AutoTokenizer.from_pretrained(model_name)

    collate_fn = partial(
        collate_finetune_with_candidates,
        pad_token_id=tok.pad_token_id,
        is_test=True,
    )

    dataloader = DataLoader(
        data_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    for batch in dataloader:
        print(batch)
        break


def _test_finetune_clm_data():
    dataloader = CNNDailyCausalLMDataModule(
        tokenizer_name="gpt2", summary_max_len=128, total_len=768
    ).train_dataloader()

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    _test_finetune_clm_data()
    # _test_finetune_seq2seq_data()
    # _test_finetune_seq2seq_with_candidates_data()
