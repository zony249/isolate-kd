import os 
import sys 

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import transformers 
from transformers import RobertaModel, RobertaConfig



class TaskFactory:
    def create_task_specific_head(task_type: str, **task_args) -> nn.Module:
        if task_type == "classification":
            return ClassificationHead(**task_args)
        else:
            raise NotImplementedError("task head not implemented")

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
            "qqp": "classification"
        }
        task_type = task_map[task]
        self.task_head = TaskFactory.create_task_specific_head(task_type, **task_args)
        return self.task_head


class RobertaTaskModel(TaskModelBase):
    def __init__(self, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        roberta_output = self.encoder(*args, **kwargs)
        task_head_output = self.task_head(roberta_output[1])
        return task_head_output 

############## Task heads ##############



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
                 pool_seq=False, 
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
            x = torch.mean(x, dim=1)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x 


if __name__ == "__main__":


    model = RobertaTaskModel(RobertaModel.from_pretrained("roberta-base"))
    model.create_task_specific_head("mnli", input_dim=768, output_dim=1)
    print(model)
    x = torch.randint(50257, size=(4, 10))
    attn_mask = torch.ones((4, 10))
    t = torch.randint(50257, size=(4, 10))
    model(input_ids=x, attention_mask=attn_mask, output_hidden_states=True)
