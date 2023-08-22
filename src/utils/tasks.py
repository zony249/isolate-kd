import os 
import sys 

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class TaskFactory:
    def create_task_specific_head(task: str, **task_args) -> nn.Module:
        if task == "classification":
            return ClassificationHead(**task_args)


class TaskHead(nn.Module):
    def __init__(self, task:str=None):
        super().__init__()
        self.task = task
    def forward(self):
        raise NotImplementedError("TaskHead is abstract, and must be subclassed") 
    
class ClassificationHead(TaskHead):
    def __init__(self, input_dim=1, output_dim=1, output_probabilities=False, *args, **kwargs):
        super().__init__(task="classification", *args, **kwargs)
        assert input_dim > 0 and output_dim > 0, "Both input_dim and output_dim have to be positive and non-zero"
        self.linear = nn.Linear(input_dim, output_dim) 
        if output_probabilities and output_dim > 1:
            self.activation = nn.Softmax()
        elif output_probabilities and output_dim == 1:
            self.activation = nn.Sigmoid() 
        else: 
            self.activation = None 
    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x 