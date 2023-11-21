from ConfigSpace import Constant, Configuration, ConfigurationSpace, Float,Integer, Categorical, Normal

from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import RandomInitialDesign
from smac.runhistory.dataclasses import TrialValue

from smac.runhistory.runhistory import RunHistory
from smac.intensifier.intensifier import Intensifier

import argparse
import subprocess
import json
import os 
import pickle 

class gpt2:
    def __init__(self, seed):
        self.seed = seed
        
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed = self.seed)
        batch_size = Integer('batch_size',(8,13),default = 12)
        block_size = Constant('block_size',1024)
        learning_rate = Float('learning_rate',(1e-6,1e-3),default = 6e-4)
        max_iters = Constant('max_iters',5)#5000)
        weight_decay = Float('weight_decay',(1e-3,1e0),default = 1e-1)
        lr_decay_iters = Constant('lr_decay_iters',5)#5000)
        seed = Constant('seed',self.seed)
        cs.add_hyperparameters([batch_size,block_size,learning_rate,max_iters,lr_decay_iters,weight_decay,seed])
        return cs


    def train(self,config:Configuration,seed:int):
    # Convert the hyperparameters to their appropriate types
        batch_size = int(config['batch_size'])
        block_size = int(config['block_size'])
        learning_rate = config['learning_rate']
        max_iters = int(config['max_iters'])
        lr_decay_iters = int(config['lr_decay_iters'])
        weight_decay = config['weight_decay']
        seed = int(config['seed'])
        

        # Generate the YAML configuration file
        yaml_config = {
            'wandb_run_name': "'gpt2-124M'",
            'batch_size': batch_size,
            'block_size': block_size,
            'learning_rate': learning_rate,
            'max_iters': max_iters,
            'lr_decay_iters': lr_decay_iters,
            'eval_interval': 1000,
            'eval_iters': 200,
            'log_interval': 10,
            'weight_decay': weight_decay,
            'seed':seed
        }


        config_content = ''.join(['{name} = {value}\n'.format(name=k, value=v) for k, v in yaml_config.items()])
        config_file_content = f'include "../default_gpt2.conf"\nconfig {{\n{config_content}\n}}'
        
        print(config_content)
        with open(f'config_files/config_{seed}.conf', 'w') as f:
            f.write(config_file_content)
        command = ['python', 'train_config.py', '-f',f'config_files/config_{seed}.conf','-c',f'seed={seed}']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()

        # Read the evaluation score from the subprocess output
        results = None
        
        os.makedirs(os.path.dirname(f'bayesian_results/results_{seed}.json'), exist_ok=True)
        with open(f'bayesian_results/results_{seed}.json') as f:
            results = json.load(f)
            # print(results)
        evaluation_score = float(results['best_val_loss'][-1])
        print(evaluation_score)

        return evaluation_score 


from pathlib import Path

def load_json_from_unknown_directory(base_directory, target_file_name):
    for dirpath, dirnames, filenames in os.walk(base_directory):
        if target_file_name in filenames:
            print(dirpath)
            return Path(dirpath)
    print('not found')
    return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-2 training with specified seed.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--index', type=int, required=True, help='Index of the SLURM array job')
    args = parser.parse_args()
    
    # Now you can use args.seed to set your seed
    seed = args.seed
    index = args.index
    
    model = gpt2(seed)
    
    state_dir = f'state_files/seed_{seed}_index_{index-1}'
    dir_path = load_json_from_unknown_directory(state_dir, 'runhistory.json')
    if dir_path is not None:
        print('is not None')
        # Load SMAC with the saved state
        #saved_configspace = Scenario.load(
        #saved_runhistory = Scenario.load(dir_path, )
        #saved_intensifier = Scenario.load(load_json_from_unknown_directory(state_dir, 'intensifier.json'))
        saved_scenario = Scenario.load(load_json_from_unknown_directory(state_dir, 'scenario.json'))
        smac = HyperparameterOptimizationFacade(
            saved_scenario,
            model.train,
            overwrite=True
        )
        print('set up: smac loaded previous',index)
    else:
        
        # Initialize SMAC for the first time
        scenario = Scenario(model.configspace, deterministic=False, output_directory=f'state_files/seed_{seed}_index_{index}', n_trials=100, seed=seed)
        initial = RandomInitialDesign(scenario, n_configs=8)
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1
        )
        smac = HyperparameterOptimizationFacade(
            scenario,
            model.train,
            intensifier=intensifier,
            initial_design=initial,
            overwrite=True
        )
        print('set up: smac 1st time')
    
    info = smac.ask()
    cost = model.train(config=info.config, seed=info.seed)
    value = TrialValue(cost=cost, time=0.5)
    smac.tell(info, value)
    
    smac.scenario.save()

