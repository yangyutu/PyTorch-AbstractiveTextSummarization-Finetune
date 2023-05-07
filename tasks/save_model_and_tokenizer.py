from module.finetune_model import FinetuneSeq2SeqModel
from transformers import (
    AutoTokenizer,
)
import argparse

def push_model_to_hub_from_ckpt(args):
    model = FinetuneSeq2SeqModel.load_from_checkpoint(args.model_ckpt).model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)

    model.save_pretrained(args.save_dir, push_to_hub=True)
    tokenizer.save_pretrained(args.save_dir, push_to_hub=True)
    
def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name", type=str, default="")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    push_model_to_hub_from_ckpt(args)