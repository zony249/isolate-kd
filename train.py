import os 
from os import path
import sys 
import time
from argparse import ArgumentParser, Namespace

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from transformers import AutoModel, AutoTokenizer

from isolate_kd.taskmodel import RobertaTaskModel, load_basic_model_and_tokenizer, TaskFactory
from isolate_kd.runner import Runner, KDRunner
from isolate_kd.environment import Env
from isolate_kd.logutils import Logger



if __name__== "__main__":
    parser = ArgumentParser(description="Runner")
    parser.add_argument("--task", type=str, choices=["wiki", "mnli"], required=True)
    parser.add_argument("--model-config", type=str, default=None, help="Optional config for model setup. "
                                                                     "If unspecified, defaults to pretrained "
                                                                     "model config")
    parser.add_argument("--data-config", type=str, default="./configs/data.yaml", help="YAML file containing dataset-"
                                                                                     "specific configurations.")
    parser.add_argument("--mode", type=str, default="finetune", choices=["finetune", "kd"])
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 mixed precision")
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--output-dir", type=str, default="./runs/", help="General output directory. Every time you"
                                                                          "run an experiment, a sub-folder will be created"
                                                                          "in here containing checkpoints and logs.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name. If you specify the same"
                                                                    "name as previously-run experiments, then"
                                                                    "training will resume from the previous"
                                                                    "experiment. Otherwise, a new experiment"
                                                                    "will be created.")
    parser.add_argument("--val-interval", type=int, default=500, help="How many steps between validations. After each validation,"
                                                                       "a checkpoint will be created.")
    parser.add_argument("--rebuild-dataset", action="store_true")

    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--teacher-path", type=str, default=None, help="Should be loadable using from_pretrained")
    parser.add_argument("--hidden-d", type=int, default=768)
    parser.add_argument("--model-d", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--kd-coeff", type=float, default=1.0) 
    parser.add_argument("--ikd-coeff", type=float, default=1.0)

    args = parser.parse_args()

    Env.set_env(task=args.task, 
                model_config=args.model_config, 
                data_config=args.data_config, 
                mode=args.mode, 
                device=args.device, 
                use_fp16=args.use_fp16,
                output_dir=args.output_dir, 
                exp_name=args.exp_name, 

                batch_size=args.batch_size, 
                epochs=args.epochs,
                lr=args.lr, 
                optim=args.optim, 
                val_interval=args.val_interval, 
                rebuild_dataset=args.rebuild_dataset,

                pretrained_ckpt=args.pretrained_ckpt,
                hidden_d=args.hidden_d, 
                model_d=args.model_d, 
                num_layers=args.num_layers,

                teacher_path=args.teacher_path,
                kd_coeff=args.kd_coeff, 
                ikd_coeff=args.ikd_coeff)

    if Env.exp_name is None:
        Env.exp_name = time.ctime(time.time()) 

    exp_dir = path.join(Env.output_dir, Env.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logfilename = path.join(exp_dir, "logfile.log")
    sys.stdout = Logger(logfilename) 
    sys.stderr = sys.stdout

    print(*sys.argv)
    Env.info()

    if Env.mode == "finetune":
        runner = Runner()
    elif "kd" in Env.mode:
        #init teacher        
        assert Env.teacher_path is not None, "For kd modes, you must specify path to teacher model"
        teacher, _ = TaskFactory.get_taskmodel(Env.task, args.teacher_path)
        teacher = teacher.to(Env.device)
        if Env.mode == "kd":
            runner = KDRunner(teacher=teacher) 
        else:
            raise NotImplementedError()
    else:
        raise ValueError()
    runner.run_loop()



