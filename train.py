from t5Pytorch import t5Pytorch
import torch
import t5
from Dataset import T5Dataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
import functools
import transformers
from nltk.translate.gleu_score import corpus_gleu
import numpy as np

from transformers import T5TokenizerFast, T5Model, T5Config


if __name__ =="__main__":
    tokenizer = T5TokenizerFast.from_pretrained('/home/nasorokin11/T5small')
    corr_dataset = T5Dataset("./heptabot-train-data/correction-train.tsv", tokenizer, "correction:", target=1, source=0)
    conll_dataset = T5Dataset("./heptabot-train-data/conll-train.tsv", tokenizer, "conll:", target=1, source=0)
    bea_dataset = T5Dataset("./heptabot-train-data/bea-train.tsv", tokenizer, "bea:", target=1, source=0)
    jfleg_dataset = T5Dataset("./heptabot-train-data/jfleg-train.tsv", tokenizer, "jfleg:", target=1, source=0)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = t5Pytorch("google/t5-v1_1-large", "/home/nasorokin11/modelslarge", torch.device("cuda:0"),
                      student=None)


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
                            ), batch_size=4,num_workers=4
                            )
    corr_dataset_eval = T5Dataset("./heptabot-train-data/correction-eval.tsv", tokenizer, "correction:", target=1,
                                  source=0)
    conll_dataset_eval = T5Dataset("./heptabot-train-data/conll-eval.tsv", tokenizer, "conll:", target=1, source=0)
    bea_dataset_eval = T5Dataset("./heptabot-train-data/bea-eval.tsv", tokenizer, "bea:", target=1, source=0)
    jfleg_dataset_eval = T5Dataset("./heptabot-train-data/jfleg-eval.tsv", tokenizer, "jfleg:", target=1, source=0)

    corr_dataloader = DataLoader(Subset(corr_dataset_eval, [i for i in range(800)]), batch_size=32, num_workers=4
                                 )
    jfleg_dataloader = DataLoader(jfleg_dataset_eval, batch_size=2, num_workers=4
                                 )
    eval_loaders = {"corr": corr_dataloader}
    #learning_rate_scheduler = functools.partial(transformers.get_constant_schedule_with_warmup, num_warmup_steps=6433
    model.train(
        dataloader,
        eval_loaders,
        steps=64336,
        save_steps=4000,
        sequence_length={"inputs": 256, "targets": 256},
        split="train",
        batch_size=4,
        optimizer=functools.partial(transformers.AdamW, lr=3e-5),
        tokenizer=tokenizer,

    )