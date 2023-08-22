import os 
import sys 

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import accelerate 
from transformers import AutoModel, AutoTokenizer

from src.utils.tasks import TaskFactory

class TaskModel:
    def __init__(self, encoder: nn.Module):
        self.encoder = encoder 

    def create_task_specific_head(self, task:str, **task_args):
        """
        @param task: One of the following: 
                        'classification' 
        @param kwargs: task specific params
        """
        # some kind of dictionary mapping
        print(task)
        print(task_args)
        print(TaskFactory.create_task_specific_head(task, **task_args))

if __name__== "__main__":
    model = TaskModel(nn.Module())
    model.create_task_specific_head(task="classification")