import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid
space =  {
    'batch_size': np.linspace(8,13,num=6,dtype=int).tolist(),                            
    'block_size': [1024],#np.linspace(800, 1200, num=100, dtype=int).tolist(),  # M
    'lr': np.logspace(np.log10(1e-06), np.log10(1e-03), num=100).tolist(),
    'weight_decay':np.logspace(np.log10(1e-03),np.log(1e0),num=100).tolist(),
    'max_iters': [5000], #np.linspace(5000, 6000, num=100, dtype=int).tolist(),
    'wandb_log':[False],
    'wandb_project':["'owt'"],
    'wandb_run_name':["'gpt2-124M'"],
    'gradient_accumulation_steps':[5*8],
    'lr_decay_iters':[5000],
    'eval_interval':[1000],
    'eval_iters':[200],
    'log_interval':[10],
}

def main(args):
    existing_pth = args.existing
    temp_pth = args.temp
    with open(existing_pth, "r") as file:
        existing_hyperparameters = file.read().splitlines()
    ps = ParameterSampler(space, n_iter=1)
    for p in ps:
        new_hp = ''.join(['{name}={value}\n'.format(name=k, value=v) for k, v in p.items()])
        print(new_hp)
    if new_hp not in existing_hyperparameters:
        with open(existing_pth,"a") as file: 
            file.write(new_hp)
        with open(temp_pth,"w") as file:
            results = ""
            results+=new_hp
            file.write(results)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description = 'Generate a hyperparameter')
    parser.add_argument('existing',type = str,
            help = 'exisiting hyperparameter file')
    parser.add_argument('temp',type = str,
            help = 'temporary hyperparameter file')
    args = parser.parse_args()
    main(args)

