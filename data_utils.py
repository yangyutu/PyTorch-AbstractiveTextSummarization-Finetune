from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer, AutoTokenizer
from functools import partial
from datasets import load_dataset
import os

# because we are performing tokenization inside each worker, we need to disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class CNNDailySeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer_name,
        split="train",
        summary_max_len=-1,
        is_test=False,
        article_max_len=512,
        model_type="",
    ):
        """data format: article, abstract, [(candidiate_i, score_i)]"""
        self.dataset_raw = load_dataset("cnn_dailymail", version="3.0.0")[split]
        self.tok = AutoTokenizer.from_pretrained(
            tokenizer_name, verbose=False, model_max_length=512
        )
        self.summary_max_len = summary_max_len
        self.is_test = is_test
        self.article_max_len = article_max_len
        self.model_type = model_type

        # pre-load the whole dataset to memory will significantly speed up
        self.dataset = {
            "article": list(self.dataset_raw["article"]),
            "highlights": list(self.dataset_raw["highlights"]),
        }

    def __len__(self):
        return len(self.dataset["article"])

    def __getitem__(self, idx):

        src_txt = self.dataset["article"][idx]
        if self.model_type.lower() == "t5":
            # Prefix the input with a prompt so T5 knows this is a summarization task.
            src_txt = "summarize: " + src_txt
        src_input_ids = self.tok(
            [src_txt],
            max_length=self.article_max_len,
            return_tensors="pt",
            pad_to_max_length=False,
            truncation=True,
        )["input_ids"].squeeze(0)

        abstract_txt = self.dataset["highlights"][idx].replace("\n", " ")
        abstract_input_ids = self.tok(
            [abstract_txt],
            max_length=self.summary_max_len,
            return_tensors="pt",
            pad_to_max_length=False,
            truncation=True,
        )["input_ids"].squeeze(0)
        # target_ids has the bos and eos tokens at the start and end
        result = {
            "src_input_ids": src_input_ids,
            "target_ids": abstract_input_ids,
        }
        if self.is_test:
            result["data"] = {
                "article": src_txt,
                "highlights": abstract_txt,
            }
        return result


class CNNDailyCausalLMDataset(Dataset):
    def __init__(
        self,
        tokenizer_name,
        split="train",
        summary_max_len=128,
        is_test=False,
        total_len=768,
        model_type="",
    ):
        self.dataset_raw = load_dataset("cnn_dailymail", version="3.0.0")[split]
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, verbose=False)
        self.summary_max_len = summary_max_len
        self.is_test = is_test
        self.total_len = total_len
        self.model_type = model_type

        # pre-load the whole dataset to memory will significantly speed up
        self.dataset = {
            "article": list(self.dataset_raw["article"]),
            "highlights": list(self.dataset_raw["highlights"]),
        }

    def __len__(self):
        return len(self.dataset["article"])

    def __getitem__(self, idx):

        src_txt = self.dataset["article"][idx]

        prompt = "\nTL;DR:\n"

        prompt_ids = self.tok.encode(prompt, add_special_tokens=False)

        src_input_ids = self.tok.encode(
            src_txt, add_special_tokens=False, truncation=True, max_length=512
        )
        abstract_txt = self.dataset["highlights"][idx].replace("\n", " ")

        abstract_input_ids = self.tok.encode(
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
            "start_idx": src_allow_len + len(prompt_ids),
        }
        if self.is_test:
            result["data"] = {
                "article": src_txt,
                "highlights": abstract_txt,
            }
        return result


def collate_finetune(batch, pad_token_id, is_test=False):
    # This collate function is mainly used for padding
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
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


def collate_clm_finetune(batch, pad_token_id=-100, is_test=False):
    # This collate function is mainly used for padding
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, : x.size(0)] = x
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


def _test_finetune_seq2seq_data():
    data_set = CNNDailySeq2SeqDataset(
        "t5-large",
        is_test=True,
        summary_max_len=512,
        article_max_len=1024,
        model_type="t5",
    )

    # for i in range(10):
    #     print(data_set[i])

    tok = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tok = AutoTokenizer.from_pretrained("t5-large")

    collate_fn = partial(
        collate_finetune,
        pad_token_id=tok.pad_token_id,
        is_test=True,
    )

    dataloader = DataLoader(
        data_set, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    for batch in dataloader:
        print(batch)
        break


def _test_finetune_clm_data():
    data_set = CNNDailyCausalLMDataset(
        "gpt2", is_test=True, summary_max_len=512, total_len=1024, model_type="gpt2"
    )

    for i in range(10):
        print(data_set[i])

    tok = AutoTokenizer.from_pretrained("gpt2")

    collate_fn = partial(
        collate_clm_finetune,
        is_test=True,
    )

    dataloader = DataLoader(
        data_set, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    _test_finetune_clm_data()
    _test_finetune_seq2seq_data()
