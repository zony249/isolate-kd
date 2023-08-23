import os 
from os import path 
import sys
import json
from tqdm import tqdm
import pickle

import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader 

from transformers import RobertaTokenizer


class DatasetBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

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

class MNLI(DatasetBase):
    def __init__(self, 
                 fpath, 
                 tokenizer, 
                 cache_to="data/cache/", 
                 rebuild_dataset=False, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs) 
        self.tok = tokenizer 
        cache_path = path.join(cache_to, "mnli.pkl")
        try: 
            # pickle load
            self.load(cache_path)
            if rebuild_dataset:
                print("Rebuilding dataset...")
                self.data = self.parse_file(fpath)
                os.makedirs(cache_to, exist_ok=True)
                self.save(cache_path)
            else: 
                print("Cached dataset loaded.")
        except FileNotFoundError:
            print(f"{cache_path} not found. Tokenizing dataset...")
            self.data = self.parse_file(fpath)
            os.makedirs(cache_to, exist_ok=True)
            self.save(cache_path)
        print(self.data[0]) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def parse_file(self, fpath: str):
        data = []
        with open(fpath, 'r') as f:
            for line in tqdm(f):
                obj = json.loads(line)
                label = obj['gold_label']
                if label == 'neutral':
                    label_id = 1 
                elif label == 'entailment':
                    label_id = 2 
                else:
                    label_id = 0 
                sent1, sent2 = obj['sentence1'].rstrip().lower(), obj['sentence2'].rstrip().lower()
                sent1_ids, sent2_ids = self.tok.encode(sent1), self.tok.encode(sent2) 
                sent_ids = sent1_ids + [self.tok.sep_token_id] + sent2_ids 
                data.append((sent_ids, label_id))
        return data 
    

if __name__ == "__main__":
    tok = RobertaTokenizer.from_pretrained("roberta-base") 
    dataset = MNLI("data/MNLI/original/multinli_1.0_train.jsonl", tok)
