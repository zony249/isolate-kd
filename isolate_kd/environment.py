from dataclasses import dataclass

import torch


@dataclass 
class Env: 
    
    task: str = "mnli"
    model_config = None
    data_config = "./configs/data.yaml" 
    device = torch.device("cuda")
    use_fp16 = False

    random_init = False
    batch_size = 16
    epochs = 1
    lr = 1e-5
    optim = "adamw"
    output_dir = "runs/"
    exp_name = "custom" 
    val_interval = 100
    rebuild_dataset = False

    
    hidden_d = None
    model_d = None
    num_layers = None

    def set_env(**kwargs):
        non_attr = []
        for k, v in kwargs.items(): 
            if hasattr(Env, k):
                setattr(Env, k, v)
            else: 
                non_attr.append(k)
        if len(non_attr) > 0:
            print("The following kwargs are not attributes of Env:", non_attr)

    def info():
        string = "Env:\n"
        for attr in vars(Env):
            string += "\t" + attr + f": {getattr(Env, attr)}\n" if (
                not attr.startswith("_") and not callable(getattr(Env, attr))
            ) else ""
        print(string)
