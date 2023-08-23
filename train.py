import os 
import sys 
from argparse import ArgumentParser, Namespace

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import AutoModel, AutoTokenizer

from src.tasks import TaskFactory




if __name__== "__main__":
    parser = ArgumentParser("Training Script")
    args = parser.parse_args()

    model = TaskModel(nn.Module())
    model.create_task_specific_head(task="classification")
