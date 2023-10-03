import os 
from os import path 
import sys
import json
from tqdm import tqdm
import pickle
import yaml
from typing import Tuple, List, Union, Dict
from copy import deepcopy

import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset

from transformers import (
    RobertaTokenizer, 
    RobertaTokenizerFast,
    AutoTokenizer, 
    PreTrainedTokenizer
)

from kdist.glue.src.tasks import MultiNLITask
from kdist.environment import Env

class DatasetBase(Dataset):
    def __init__(self, 
                 split="train", 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs) 
        self.split = split

    def parse_file(self, fpath: str):
        raise NotImplementedError 
    
    def load(self, fpath:str):
        with open(fpath, 'rb') as f:
            temp_dict = pickle.load(f)
        self.__dict__.clear()
        self.__dict__.update(temp_dict)

    def save(self, fpath:str):
        with open(fpath, 'wb') as f: 
            pickle.dump(self.__dict__, f)

    def set_split(self, split:str) -> Dataset: 
        assert split == "train" or split == "val" or split == "dev",\
            "split can either be train, or val / dev"
        self.split = split
        return self
    
    def commit(self) -> Dataset:
        """
        commits the split, i.e. throws away the other split to save
        memory 
        """
        if self.split == "train":
            self.val_data = None 
            self.val_data_text = None 
        else:
            self.train_data = None
            self.train_data_text = None 
        return self

    def has_val(self):
        return self.val_data is not None


class MultiNLI(MultiNLITask, DatasetBase):
    def __init__(self, fpath: str, 
                 max_seq_len: int, 
                 tokenizer: PreTrainedTokenizer,
                 cache_to: str = "data/cache/", 
                 rebuild_dataset: bool = False,
                 *args, **kwargs):
        MultiNLITask.__init__(self, fpath, max_seq_len, *args, **kwargs)
        DatasetBase.__init__(self, *args, **kwargs)
        self.tokenizer = tokenizer 
        self.cache_to = cache_to 
        
        os.makedirs(cache_to, exist_ok=True) 
        self.savepath = path.join(cache_to, self.name + ".pkl") 
        try:
            if rebuild_dataset:
                print("Rebuilding dataset...")
                self.load_data(fpath, max_seq_len)
                self.train_data = self.tokenize_sents(self.train_data_text)
                self.val_data = self.tokenize_sents(self.val_data_text)
                self.save(self.savepath)
                print(f"Saved dataset to {self.savepath}.")
            else: 
                print(f"Loading cached dataset from {self.savepath}")
                self.load(self.savepath)
                print(f"Loaded cached dataset from {self.savepath}!")
        except FileNotFoundError:
            print("Cached file not found. Building dataset...")
            self.load_data(fpath, max_seq_len)
            self.train_data = self.tokenize_sents(self.train_data_text)
            self.val_data = self.tokenize_sents(self.val_data_text)
            self.save(self.savepath)
            print(f"Saved dataset to {self.savepath}.")
        print(self.train_data[0])

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.train_data)
        else:
            return len(self.val_data)

    def __getitem__(self, index) -> Tuple:
        if self.split == "train":
            return (torch.tensor(self.train_data[index][0]), 
                    torch.tensor(self.train_data[index][1]), 
                    self.train_data[index][2])
        else:
            return (torch.tensor(self.val_data[index][0]), 
                    torch.tensor(self.val_data[index][1]), 
                    self.val_data[index][2])

    def tokenize_sents(self, cluster: Tuple[List, List, List]):
        """
        @param cluster: (sent1, sent2, classes)
        """
        processed = []
        for sent1, sent2, cls in tqdm(zip(cluster[0], cluster[1], cluster[2]), desc="Tokenizing dataset..."):
            sent1_ids = self.tokenizer.encode(sent1)
            sent2_ids = self.tokenizer.encode(sent2)
            sent_ids = sent1_ids + [self.tokenizer.sep_token_id] + sent2_ids
            attn_mask = [1 for _ in range(len(sent_ids))]
            processed.append((
                sent_ids, 
                attn_mask, 
                cls))
        return processed 
    

    def pad_collate(self, batch):
        (xx, aa, yy) = zip(*batch)

        xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
        aa_pad = torch.nn.utils.rnn.pad_sequence(aa, batch_first=True, padding_value=0)
        y_batch = torch.tensor(yy)

        return xx_pad, aa_pad, y_batch 
        

class Wikipedia(DatasetBase):
    def __init__(self,
                 max_seq_len: int, 
                 tokenizer: PreTrainedTokenizer, 
                 cache_to: str = "data/cache/", 
                 rebuild_dataset: bool = False, 
                 *args, 
                 **kwargs):
        self.name = "wiki_en"
        os.makedirs(cache_to, exist_ok=True)
        self.savepath = path.join(cache_to, self.name + ".pkl")
        super().__init__(*args, **kwargs)
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        try:
            if rebuild_dataset:
                print("Rebuilding dataset...")
                self.train_data = load_dataset("wikipedia", "20220301.en", cache_dir="data")["train"]
                self.val_data = None
                self.save(self.savepath)
                print(f"Saved dataset to {self.savepath}")
            else: 
                print(f"Loading cached dataset from {self.savepath}")
                self.load(self.savepath)
                print(f"Loaded cached dataset from {self.savepath}!")
        except FileNotFoundError:
            print("Cached file not found. Building dataset...")
            self.train_data = load_dataset("wikipedia", "20220301.en", cache_dir="data")["train"]
            self.val_data = None
            self.save(self.savepath)
            print(f"Saved dataset to {self.savepath}.")
        # print(self.train_data[0])

    def __getitem__(self, idx):
        stringitem = self.train_data[idx]["text"] 
        tokenized = self.tokenizer(stringitem, truncation=True, max_length=self.max_seq_len) 
        labels = torch.tensor(tokenized["input_ids"])
        input_ids = deepcopy(labels)
        indexer = torch.zeros(len(input_ids), dtype=torch.long)
        indexer[:int(0.15 * len(indexer))] = 1 
        indexer = indexer[torch.randperm(len(indexer))]
        input_ids = input_ids.masked_fill_(indexer, self.tokenizer.mask_token_id)
        attn_mask = torch.tensor(tokenized["attention_mask"])
        
        return input_ids, attn_mask, labels

    def __len__(self):
        return len(self.train_data)

    def pad_collate(self, batch):
        (xx, aa, yy) = zip(*batch)

        xx_pad = torch.nn.utils.rnn.pad_sequence(xx, 
                                                 batch_first=True, 
                                                 padding_value=self.tokenizer.pad_token_id)
        aa_pad = torch.nn.utils.rnn.pad_sequence(aa, 
                                                 batch_first=True, 
                                                 padding_value=0)
        yy_pad = torch.nn.utils.rnn.pad_sequence(yy, 
                                                 batch_first=True, 
                                                 padding_value=self.tokenizer.pad_token_id)
        return xx_pad, aa_pad, yy_pad 

if __name__ == "__main__":
    tok = RobertaTokenizerFast.from_pretrained("roberta-base") 
    # dataset = MNLI("data/MNLI/original/multinli_1.0_train.jsonl", tok)

    with open(Env.data_config, 'r') as f:
        yaml_obj = yaml.safe_load(f)
    print(yaml_obj["mnli"]["base"]) 
    # dataset = MultiNLI(yaml_obj["mnli"]["base"], 128, tokenizer=tok, rebuild_dataset=False)
    dataset = Wikipedia(256, tok, rebuild_dataset=True) 
    # print(dataset.train_data_text)
    
    dataset[0]

    dataloader = DataLoader(dataset, 
                            batch_size=8, 
                            shuffle=True, 
                            collate_fn=dataset.pad_collate)
    dataloader = iter(dataloader)
    print(next(dataloader))
    print(next(dataloader))
    print(next(dataloader))
    print(next(dataloader))
    print(next(dataloader))
    print(next(dataloader))

    for i in tqdm(range(1000), desc="Timing"):
        next(dataloader)
