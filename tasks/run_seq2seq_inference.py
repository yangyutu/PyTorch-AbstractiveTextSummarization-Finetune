import argparse
from data.data_utils import CNNDailySeq2SeqDataModule
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import tqdm
from transformers import pipeline
import torch
import evaluate


def main(args):
    # fix random seeds for reproducibility
    torch_dtype = torch.float32 if args.precision == 32 else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name, torch_dtype=torch_dtype)
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

    rouge = evaluate.load("rouge")
    score = rouge.compute(predictions=final_outputs, references=references)
    print(score)


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=int, choices=[32, 16], default=32)
    parser.add_argument("--article_truncate", type=int, default=512)
    parser.add_argument("--summary_truncate", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument(
        "--pretrained_model_name", type=str, default="facebook/bart-large-cnn"
    )
    parser.add_argument("--add_prefix", type=str, default="")
    parser.add_argument("--add_suffix", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
