import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid
space =  {
    'batch_size': [8,13],
    'block_size': [1024],
    'lr': np.logspace(np.log10(1e-06), np.log10(1e-03), num=100).tolist(),
    'weight_decay':np.linspace(1e-03,1e-0,num=100).tolist(),
    'max_iters':[30],#5000],#$np.linspace(5000, 6000, num=100, dtype=int).tolist(),
    'wandb_log':[False],
    'wandb_project':["'owt'"],
    'wandb_run_name':["'gpt2-124M'"],
    'gradient_accumulation_steps':[5*8],
    'lr_decay_iters':[30],#5000],
    'eval_interval':[1000],
    'eval_iters':[200],
    'log_interval':[10],
}
def adjust_params(batch_size, block_size, max_product):
    """Adjusts batch_size and block_size to ensure their product is under max_product."""
    if batch_size * block_size < max_product:
        return batch_size, block_size
    else:
        # Decide which parameter to adjust
        if batch_size > block_size:
            batch_size = max_product // block_size
        else:
            block_size = max_product // batch_size
        # Ensure the product is under max_product after adjustment
        return adjust_params(batch_size, block_size, max_product)

def main(args):
    existing_pth = args.existing
    temp_pth = args.temp

    with open(existing_pth, "r") as file:
        existing_hyperparameters = file.read().splitlines()

    # Iterate over all combinations in the parameter space
    for batch_size in space['batch_size']:
        for block_size in space['block_size']:
            # Adjust parameters if needed
            adjusted_batch_size, adjusted_block_size = adjust_params(batch_size, block_size, 12500)
            
            # Update your parameter space with the adjusted values
            updated_space = space.copy()
            updated_space['batch_size'] = [adjusted_batch_size]
            updated_space['block_size'] = [adjusted_block_size]

            # Sample parameters
            ps = ParameterSampler(updated_space, n_iter=1)
            for p in ps:
                new_hp = ''.join(['{name}={value}\n'.format(name=k, value=v) for k, v in p.items()])
                
                if new_hp not in existing_hyperparameters:
                    with open(existing_pth, "a") as file:
                        file.write(new_hp)
                    with open(temp_pth, "w") as file:
                        file.write(new_hp)

# Command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter generation with constraints')
    parser.add_argument('existing', type=str, help='Path to the existing hyperparameter file')
    parser.add_argument('temp', type=str, help='Path to the temporary hyperparameter file')
    args = parser.parse_args()
    main(args)
