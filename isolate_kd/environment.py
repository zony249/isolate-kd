from dataclasses import dataclass

import torch


@dataclass 
class Env: 

    # general 
    task: str = "mnli"
    model_config = None
    data_config = "./configs/data.yaml" 
    mode = "finetune"
    device = torch.device("cuda")
    use_fp16 = False
    output_dir = "runs/"
    exp_name = "custom" 

    # Training hyperparameters
    batch_size = 16
    epochs = 1
    lr = 1e-5
    optim = "adamw"
    output_dir = "runs/"
    rebuild_dataset = False
    val_interval=100

    # model specific
    hidden_d = None
    model_d = None
    num_layers = None
    pretrained_ckpt = None

    #kd arguments
    teacher_path=None 
    kd_coeff=1.0
    ikd_coeff=1.0



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
