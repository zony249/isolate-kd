import os
import sys 
from copy import deepcopy
from os import path
import time
import pickle
from tqdm import tqdm, trange
import logging
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset 
from torch.optim import AdamW, Adam, SGD
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

from isolate_kd.taskmodel import TaskFactory
from isolate_kd.environment import Env
from isolate_kd.logutils import Logger

class Runner: 
    def __init__(self):
        self.task = Env.task
        self.device = Env.device

        self.model, self.tok = TaskFactory.get_new_taskmodel(self.task)
        self.model = self.model.to(self.device)
        self.optim = self.create_optim(self.model)        
        self.scaler = GradScaler() 
        self.step_id = 0
        self.epoch_id = 0
        self.total_steps = 0
        self.val_interval = Env.val_interval

        trainset = TaskFactory.get_dataset(self.task, self.tok, Env.rebuild_dataset)
        valset = deepcopy(trainset)

        # Clean up dataset by getting rid of the other split
        self.valset = valset.set_split("val").commit()
        self.trainset = trainset.set_split("train").commit()

        self.tloader = DataLoader(trainset, 
                            batch_size=Env.batch_size, 
                            shuffle=True, 
                            collate_fn=trainset.pad_collate,
                            num_workers=4)
        if self.valset.has_val():
            self.vloader = DataLoader(valset, 
                                batch_size=Env.batch_size, 
                                shuffle=True, 
                                collate_fn=valset.pad_collate)
        pathname = path.join(Env.output_dir, Env.exp_name)

        try:
            # load state and continue training 
            (optim_state_dict, 
            scaler_state_dict,
            self.step_id, 
            self.epoch_id, 
            self.total_steps) = self.load_train_state(path.join(pathname, "trainstate.pt"))
            self.optim.load_state_dict(optim_state_dict)
            self.scaler.load_state_dict(scaler_state_dict)
            self.model.from_pretrained(pathname)
            self.model = self.model.to(self.device)
            print(f"Continuing training from epoch {self.epoch_id}, step {self.step_id}.")
        except:
            print("Training start!")
        self.model.save_pretrained(pathname)
        self.save_train_state(path.join(pathname, "optim.pkl"))
        print(self.model)

        self.start_epoch = self.epoch_id
    
    def run_loop(self):
        end_epoch = self.start_epoch + Env.epochs
        for epoch in range(self.start_epoch, self.start_epoch + Env.epochs):
            self.epoch_id = epoch
            tbar = trange(
                self.step_id, 
                len(self.trainset)//Env.batch_size, 
                desc=f"Training epoch {epoch}/{end_epoch}", 
                initial=self.step_id)
            iter_tloader = iter(self.tloader)
            for step in tbar:
                batch = next(iter_tloader)
                loss = self.train_forward(batch, step)
                tbar.set_postfix(loss=f"{loss:.4f}", refresh=False)
                if step % self.val_interval == 0:
                    metric = self.run_val()
                    self.set_checkpoint(path.join(Env.output_dir, Env.exp_name), 
                                        self.model)
            self.step_id = 0
    def forward_pass(self, model, input_ids, attention_mask, labels):
        if Env.use_fp16:
            with torch.autocast(self.device):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=False)
                y = outputs.logits
                loss = F.cross_entropy(y.reshape(-1, y.shape[-1]), labels.view(-1))
                outputs.loss = loss
        else:
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=False)
            y = outputs.logits
            loss = F.cross_entropy(y.reshape(-1, y.shape[-1]), labels.view(-1))
            outputs.loss = loss
        return outputs
    def backward_pass_and_step(self, loss, optim):
        if Env.use_fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(optim)
            self.scaler.update()
        else:
            loss.backward()
            optim.step()

    ## TODO
    def train_forward(self, batch, step_id): 
        self.model.train() 
        x, attn_mask, t = batch 
        x = x.to(self.device)
        attn_mask = attn_mask.to(self.device)
        t = t.to(self.device)
        outputs = self.forward_pass(self.model, x, attn_mask, t)
        self.backward_pass_and_step(outputs.loss, self.optim)
        self.optim.zero_grad(set_to_none=True)
        self.step_id = step_id
        self.total_steps += 1
        return outputs.loss.item()

    def run_val(self):
        self.model.eval()
        dummy = torch.ones((1, 3), dtype=torch.long, device=self.device)
        # print(self.model(input_ids=dummy, attention_mask=dummy).logits)
        if not self.valset.has_val():
            return 0
        vbar = trange(0, len(self.valset)//Env.batch_size, 
            desc=f"Validating") 
        iter_vloader = iter(self.vloader)
        loss_accum = []
        for step in vbar:
            batch = next(iter_vloader)
            loss = self.val_forward(batch, step)
            vbar.set_postfix(loss=f"{loss:.4f}", refresh=False)
            loss_accum.append(loss)
        return np.mean(loss_accum)
         
    def val_forward(self):
        pass
    
    def create_optim(self, model): 
        if Env.optim == "adamw":
            optim = AdamW(model.parameters(), lr=Env.lr)
        elif Env.optim == "adam":
            optim = Adam(model.parameters(), lr=Env.lr)
        elif Env.optim == "sgd":
            optim = SGD(model.parameters(), lr=Env.lr, momentum=0.9)
        else: 
            raise NotImplementedError
        return optim
    
    def save_train_state(self, fpath):
        torch.save({
            "optim": self.optim.state_dict(), 
            "scaler": self.scaler.state_dict(), 
            "step_id": self.step_id, 
            "epoch_id": self.epoch_id, 
            "total_steps": self.total_steps
            }, fpath)

    def load_train_state(self, fpath):
        ckpt = torch.load(fpath)
        optim_state = ckpt["optim"]
        scaler_state = ckpt["scaler"]
        step_id = ckpt["step_id"]
        epoch_id = ckpt["epoch_id"]
        total_steps = ckpt["total_steps"]
        return optim_state, scaler_state, step_id, epoch_id, total_steps 

    def set_checkpoint(self, 
                        exp_path: str, 
                        model):
        model.save_pretrained(exp_path)
        self.save_train_state(path.join(exp_path, "trainstate.pt"))
        print(f"checkpoint set: step_id: {self.step_id}, epoch_id: {self.epoch_id}, total_steps: {self.total_steps}")
        
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Runner")
    parser.add_argument("--task", type=str, choices=["wiki", "mnli"], required=True)
    parser.add_argument("--model-config", type=str, default=None, help="Optional config for model setup. "
                                                                     "If unspecified, defaults to pretrained "
                                                                     "model config")
    parser.add_argument("--data-config", type=str, default="./configs/data.yaml", help="YAML file containing dataset-"
                                                                                     "specific configurations.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training.")
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--random-init", action="store_true", help="Randomly initialize the model.")
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
    parser.add_argument("--val-interval", type=str, default=500, help="How many steps between validations. After each validation,"
                                                                       "a checkpoint will be created.")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--hidden-d", type=int, default=768)
    parser.add_argument("--model-d", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)

    args = parser.parse_args()

    Env.set_env(task=args.task, 
                model_config=args.model_config, 
                data_config=args.data_config, 
                device=args.device, 
                use_fp16=args.use_fp16,
                random_init=args.random_init, 
                batch_size=args.batch_size, 
                epochs=args.epochs,
                lr=args.lr, 
                optim=args.optim, 
                output_dir=args.output_dir, 
                exp_name=args.exp_name, 
                val_interval=args.val_interval, 
                rebuild_dataset=args.rebuild_dataset,
                hidden_d=args.hidden_d, 
                model_d=args.model_d, 
                num_layers=args.num_layers)

    if Env.exp_name is None:
        Env.exp_name = time.ctime(time.time()) 

    exp_dir = path.join(Env.output_dir, Env.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logfilename = path.join(exp_dir, "logfile.log")
    sys.stdout = Logger(logfilename) 
    sys.stderr = sys.stdout

    print(*sys.argv)
    Env.info()

    runner = Runner()
    runner.run_loop()

