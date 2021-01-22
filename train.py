from t5Pytorch import t5Pytorch
import torch
from Dataset import T5Dataset, JflegDataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
import functools
import transformers
from argparse import ArgumentParser
import os
import sys

from transformers import T5TokenizerFast, T5Model, T5Config


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--path", "-n", help="path to datasets folder",
                        default="baseline")
    parser.add_argument("--dir", "-d", default=None, help="directory for saving files")
    parser.add_argument("--tokenizer", "-t", help="path to tokenizer")
    parser.add_argument("--model", help="model")
    parser.add_argument("--student", help="student model")
    return parser.parse_args()


def main(args):
    tokenizer = T5TokenizerFast.from_pretrained('/home/nasorokin11/T5small')
    corr_dataset = T5Dataset(os.path.join(args.path, "correction-train.tsv"), tokenizer, "correction:", target=1, source=0)
    conll_dataset = T5Dataset(os.path.join(args.path, "conll-train.tsv"), tokenizer, "conll:", target=1, source=0)
    bea_dataset = T5Dataset(os.path.join(args.path, "bea-cut-train.tsv"), tokenizer, "bea:", target=1, source=0)
    jfleg_dataset = T5Dataset(os.path.join(args.path, "jfleg-train.tsv"), tokenizer, "jfleg:", target=1, source=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = t5Pytorch(args.model, args.dir, device,
                      student=args.student)

    def temperature_to_weights(dataset_lengths, temperature=2.0, maximum=None, scale=1.0):
        '''Calculate mixing rates'''
        mixing_rates = []
        for length in dataset_lengths:
            rate = length * scale
            if maximum:
                rate = min(rate, maximum)
            if temperature != 1.0:
                rate = rate ** (1.0 / temperature)
            mixing_rates.append(rate)
        return mixing_rates

    datasets = [corr_dataset, conll_dataset, bea_dataset, jfleg_dataset]
    dataset_lengths = [len(d) for d in datasets]
    dataset_weights = temperature_to_weights(dataset_lengths)

    # Calculate weights per sample
    weights = []
    for i in range(len(datasets)):
        weights += [dataset_weights[i]] * len(datasets[i])

    dataloader = DataLoader(ConcatDataset(datasets),
                            sampler=WeightedRandomSampler(
                                num_samples=min(dataset_lengths),
                                weights=weights,
                            ), batch_size=4, num_workers=4
                            )
    corr_dataset_eval = T5Dataset(os.path.join(args.path, "correction-eval.tsv"), tokenizer, "correction:", target=1,
                                  source=0)
    conll_dataset_eval = T5Dataset(os.path.join(args.path, "conll-eval.tsv"), tokenizer, "conll:", target=1, source=0)
    bea_dataset_eval = T5Dataset(os.path.join(args.path, "bea-cut-eval.tsv"), tokenizer, "bea:", target=1, source=0)
    jfleg_dataset_eval = T5Dataset(os.path.join(args.path,"jfleg-eval.tsv"), tokenizer, "jfleg:", target=1, source=0)

    corr_dataloader = DataLoader(corr_dataset_eval, batch_size=64, num_workers=4
                                 )
    jfleg_dataloader = DataLoader(jfleg_dataset_eval, batch_size=64, num_workers=4
                                  )
    bea_dataloader = DataLoader(bea_dataset_eval, batch_size=2, num_workers=4
                                )
    eval_loaders = {"corr": corr_dataloader, "bea": bea_dataloader}
    # learning_rate_scheduler = functools.partial(transformers.get_constant_schedule_with_warmup, num_warmup_steps=6433
    model.train(
        dataloader,
        eval_loaders,
        steps=62336,
        save_steps=2000,
        sequence_length={"inputs": 256, "targets": 256},
        split="train",
        batch_size=4,
        optimizer=functools.partial(transformers.AdamW, lr=3e-5),
        tokenizer=tokenizer,
        learning_rate_scheduler=functools.partial(transformers.get_constant_schedule_with_warmup, num_warmup_steps=6433)
    )


if __name__ =="__main__":
    args = parse_arguments()
    sys.exit(main(args))