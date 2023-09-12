import os 
import sys 
import yaml 
from os import path
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass, field 

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import transformers 
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from isolate_kd.data import DatasetBase, MultiNLI, Wikipedia
from isolate_kd.environment import Env

class TaskModelBase(nn.Module):
    def __init__(self, 
                 encoder: nn.Module, 
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder 
        self.task_head = None

    def forward(self, x):
        if self.task_head is None:
            raise NotImplementedError("Task Head is None. Please run self.create_task_specific_head to create one.")
        raise NotImplementedError("This class is abstract. The forward function defines how data passes through \
                                  the encoder and goes to the task head")

    def create_task_specific_head(self, task:str, **task_args):
        """
        @param task: One of the following: 
                        'classification' 
        @param kwargs: task specific params
        """
        task_map = {
            "mnli": "classification", 
            "qqp": "classification", 
            "wiki": "mlm"
        }
        task_type = task_map[task]
        self.task_head = TaskFactory.create_task_specific_head(task_type, **task_args)
        return self.task_head

    def from_pretrained(self, fpath):
        self.encoder = self.encoder.from_pretrained(fpath)
        self.task_head.load_state_dict(torch.load(path.join(fpath, "taskhead.pt")))
        return self 

    def save_pretrained(self, fpath):
        self.encoder.save_pretrained(fpath)
        torch.save(self.task_head.state_dict(), path.join(fpath, "taskhead.pt"))

class RobertaTaskModel(TaskModelBase):
    def __init__(self, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        roberta_output = self.encoder(*args, **kwargs)
        task_head_output = self.task_head(roberta_output[0])
        return MLMOutput(
            logits=task_head_output,
            hidden_states=roberta_output.hidden_states,
            attentions=roberta_output.attentions
        ) 


############## FUNCTIONS #################

class TaskFactory:
    def create_task_specific_head(task_type: str, **task_args) -> nn.Module:
        if task_type == "classification":
            return ClassificationHead(**task_args)
        elif task_type == "mlm":
            return MLMHead(**task_args)
        else:
            raise NotImplementedError("task head not implemented")

    def get_new_taskmodel(task:str) -> Tuple[TaskModelBase, PreTrainedTokenizer]:
        if task == "mnli":
            enc, tok = load_basic_model_and_tokenizer("roberta-base")
            taskmodel = RobertaTaskModel(enc)
            taskmodel.create_task_specific_head(task, input_dim=enc.config.hidden_size, output_dim=3)
            return taskmodel, tok
        elif "wiki" in task:
            enc, tok = load_basic_model_and_tokenizer("roberta-base")
            taskmodel = RobertaTaskModel(enc)
            taskmodel.create_task_specific_head(task, vocab_size=enc.config.vocab_size, hidden_d=enc.config.hidden_size)
            return taskmodel, tok
        else:
            raise NotImplementedError(f"taskmodel not implemented for {task}")
        

    def get_dataset(task:str, 
                    tok:PreTrainedTokenizer, 
                    rebuild:bool=False) -> DatasetBase: 
        if task == "mnli": 
            with open(Env.data_config, 'r') as f:
                yaml_obj = yaml.safe_load(f)
            dataset = MultiNLI(yaml_obj["mnli"]["base"], 128, tokenizer=tok, rebuild_dataset=False)
            return dataset
        elif "wiki" in task:
            return Wikipedia(
                max_seq_len=256, 
                tokenizer=tok, 
                rebuild_dataset=rebuild
            )
        else:
            raise NotImplementedError(f"get_dataset is not implemented for {task}")

def load_basic_model_and_tokenizer(name_or_path:str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModel.from_pretrained(name_or_path)
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if Env.random_init: 
        config = model.config
        if Env.hidden_d is not None:
            setattr(config, translate_attr(config.model_type, "hidden_d"), Env.hidden_d)
        if Env.model_d is not None: 
            setattr(config, translate_attr(config.model_type, "model_d"), Env.model_d)
        if Env.num_layers is not None: 
            setattr(config, translate_attr(config.model_type, "num_hidden_layers"), Env.num_layers)
        if Env.model_config is not None:
            config = Env.model_config
        model = AutoModel.from_config(config)
    return model, tok

def translate_attr(model_type:str, attr_from:str):
    """
    translates roberta config attributes model_d, 
    hidden_d, num_hidden_layers to gpt2 and t5 attributes
    """
    if "gpt" in model_type: 
        mapdict = {
            "hidden_d": "n_embd", 
            "model_d": "n_inner", 
            "num_hidden_layers": "n_layer"}
        return mapdict[attr_from]
    elif "roberta" in model_type:
        mapdict = {
            "hidden_d": "hidden_size", 
            "model_d": "intermediate_size", 
            "num_hidden_layers": "num_hidden_layers"}
        return mapdict[attr_from] 
    elif "t5" in model_type:
        mapdict = {
            "hidden_d": "d_model", 
            "model_d": "d_ff", 
            "num_hidden_layers": "num_layers"}
    else:
        raise NotImplementedError()


############## Task heads and outputs ##############

@dataclass 
class MLMOutput:
    loss:torch.Tensor=field(repr=False, default=None) 
    logits:torch.Tensor=field(repr=False, default=None) 
    hidden_states:Tuple[torch.Tensor]=field(repr=False, default=None)
    attentions:Tuple[torch.Tensor]=field(repr=False, default=None)


class TaskHead(nn.Module):
    def __init__(self, task:str=None):
        super().__init__()
        self.task = task
    def forward(self):
        raise NotImplementedError("TaskHead is abstract, and must be subclassed") 
    
class ClassificationHead(TaskHead):
    def __init__(self, 
                 input_dim=1, 
                 output_dim=1, 
                 output_probabilities=False, 
                 pool_seq=True, 
                 *args, 
                 **kwargs):
        super().__init__(task="classification", *args, **kwargs)
        self.pool_seq = pool_seq
        assert input_dim > 0 and output_dim > 0, "Both input_dim and output_dim have to be positive and non-zero"

        self.linear = nn.Linear(input_dim, output_dim) 
        if output_probabilities and output_dim > 1:
            self.activation = nn.Softmax()
        elif output_probabilities and output_dim == 1:
            self.activation = nn.Sigmoid() 
        else: 
            self.activation = None 

    def forward(self, x):
        if self.pool_seq:
            x = x[:, 0]
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x 

class MLMHead(TaskHead): 
    def __init__(self, 
                 vocab_size=None, 
                 hidden_d=768, 
                 output_probabilities=False, 
                 *args, 
                 **kwargs):
        assert vocab_size is not None, "Vocab size must be defined."
        super().__init__(task="mlm", *args, **kwargs) 
        self.layer = nn.Linear(hidden_d, vocab_size)
        if output_probabilities:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = None

    def forward(self, x):
        x = self.layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x 


################# OUTPUTS ##################

if __name__ == "__main__":


    model = RobertaTaskModel(AutoModel.from_pretrained("roberta-base"))
    model.create_task_specific_head("mnli", input_dim=768, output_dim=1)
    print(model)
    x = torch.randint(50257, size=(4, 10))
    attn_mask = torch.ones((4, 10))
    t = torch.randint(50257, size=(4, 10))
    model(input_ids=x, attention_mask=attn_mask, output_hidden_states=True)

    TaskFactory.get_dataset("mnli", AutoTokenizer.from_pretrained("roberta-base"))
