import os
import re
import pandas as pd
from torch.utils.data import Dataset
from transformers import T5TokenizerFast


def normalize_text(text):
    """Remove quotes from a TensorFlow string."""
    text = re.sub("'(.*)'", r"\1", text)
    return text


class T5Dataset(Dataset):
    def __init__(self, path, tokenizer: T5TokenizerFast, prefix, max_length=256,
                 max_target_length=256, target=1, source=0):
        self.corr = pd.read_csv(path, sep="\t", header=None)
        self.target = target
        self.source = source
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.prefix = prefix

        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        input_text = normalize_text(self.prefix + " " + self.corr[self.source][idx])
        target_text = normalize_text(self.corr[self.target][idx])
        input = self.tokenizer.prepare_seq2seq_batch([input_text], [target_text], self.max_length,
                                                     self.max_target_length, return_tensors="pt",
                                                     truncation=True, padding='max_length')

        source_ids = input["input_ids"].squeeze()
        target_ids = input["labels"].squeeze()

        src_mask = input["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    def __len__(self):
        return len(self.corr)


class JflegDataset(Dataset):
    def __init__(self, path, tokenizer: T5TokenizerFast, prefix, max_length=256,
                 max_target_length=256, target=1, source=0):
        self.corr = pd.read_csv(path)
        self.target = target
        self.source = source
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.prefix = prefix

        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        input_text = normalize_text(self.prefix + " " + self.corr["inputs"][idx])
        ref_0 = normalize_text(self.corr["ref0"][idx])
        ref_1 = normalize_text(self.corr["ref1"][idx])
        ref_2 = normalize_text(self.corr["ref2"][idx])
        ref_3 = normalize_text(self.corr["ref3"][idx])
        input = self.tokenizer.encode_plus(input_text, max_length=self.max_length, return_tensors="pt",
                                                     truncation=True, padding='max_length')
        ref_0 = self.tokenizer.encode_plus(ref_0, max_length=self.max_length, return_tensors="pt",
                                                 truncation=True, padding='max_length')
        ref_1 = self.tokenizer.encode_plus(ref_1, max_length=self.max_length, return_tensors="pt",
                                                 truncation=True, padding='max_length')
        ref_2 = self.tokenizer.encode_plus(ref_2, max_length=self.max_length, return_tensors="pt",
                                                 truncation=True, padding='max_length')
        ref_3 = self.tokenizer.encode_plus(ref_3, max_length=self.max_length, return_tensors="pt",
                                                 truncation=True, padding='max_length')
        source_ids = input["input_ids"].squeeze()
        src_mask = input["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "ref_0": ref_0["input_ids"],
               "ref_1": ref_1["input_ids"], "ref_2": ref_2["input_ids"], "ref_3": ref_3["input_ids"]}

    def __len__(self):
        return len(self.corr)

