from argparse import ArgumentParser
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch
import sys


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--path", "-n", help="path to model",
                        default="baseline")
    parser.add_argument("--data", "-d", default=None, help="string")
    parser.add_argument("--tokenizer", "-t", help="path to tokenizer")
    parser.add_argument("--type", help="Model type")
    parser.add_argument("--device", help="device")
    return parser.parse_args()


def main(args):
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer)
    input = tokenizer.encode_plus(args.data)
    source_ids = input["input_ids"].squeeze()
    model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
    model.load_state_dict(torch.load(args.path))
    model.to(args.device)

    src_mask = input["attention_mask"].squeeze()
    generated_ids = model.generate(
        input_ids=source_ids,
        attention_mask=src_mask,
        max_length=256,
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
