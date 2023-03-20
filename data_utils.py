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

def collate_finetune_with_candidates(batch, pad_token_id, is_test=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
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


class CNNDailyWithCandidatesSeq2SeqDataset(Dataset):
    def __init__(
        self,
        fdir,
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
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
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
            with open(os.path.join(self.fdir, "%d.json" % idx), "r") as f:
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
    data_set = CNNDailySeq2SeqDataset(
        "t5-large",
        is_test=True,
        summary_max_len=512,
        article_max_len=1024,
        model_type="t5",
    )

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


def _test_finetune_seq2seq_with_candidates_data():
    model_name = "facebook/bart-large-cnn"
    fdir = "/mnt/d/MLData/data/summarization/cnndm_bart/cnndm/diverse/test/"
    data_set = CNNDailyWithCandidatesSeq2SeqDataset(
        fdir=fdir,
        tokenizer_name=model_name,
        is_test=True,
        summary_max_len=128,
        article_max_len=512,
        max_cand_num=16
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
    # _test_finetune_clm_data()
    # _test_finetune_seq2seq_data()
    _test_finetune_seq2seq_with_candidates_data()
