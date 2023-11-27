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
        max_iters = Constant('max_iters',10)#5000)
        weight_decay = Float('weight_decay',(1e-3,1e0),default = 1e-1)
        lr_decay_iters = Constant('lr_decay_iters',10)#5000)
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
"""
def load_json_from_unknown_directory(base_directory, target_file_name):
    for dirpath, dirnames, filenames in os.walk(base_directory):
        if target_file_name in filenames:
            print(dirpath)
            return Path(dirpath)
    print('not found')
    return None
"""
def find_newest_directory(base_directory):
    base_path = Path(base_directory)
    directories = [d for d in base_path.iterdir() if d.is_dir()]

    if not directories:
        return None

    newest_dir = max(directories, key=lambda d: d.stat().st_ctime)
    print(newest_dir)
    return newest_dir


def load_state(state_dir, iteration, seed):
    initial_file = os.path.join(state_dir, f'initial_state_{iteration-1}.pkl')
    
    if os.path.exists(initial_file):
        with open(initial_file, 'rb') as f:
            initial = pickle.load(f)
        print('reloaded initial')
        scenario = Scenario.load(find_newest_directory(state_dir)/f'{seed}')
        return initial, scenario
    else:
        return None, None
    
def save_state(initial, state_dir, iteration):
    with open(os.path.join(state_dir, f'initial_state_{iteration}.pkl'), 'wb') as f:
        pickle.dump(initial, f)
    
def verify_loaded_state(original_smac, loaded_smac, original_scenario, loaded_scenario):
    # Implement custom verification logic here
    # Example:
    if original_smac.runhistory != loaded_smac.runhistory:
        print("Warning: Run histories do not match.")
    if original_scenario != loaded_scenario:
        print("Warning: Scenarios do not match.")
    # Add other necessary checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-2 training with specified seed.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--index', type=int, required=True, help='Index of the SLURM array job')
    args = parser.parse_args()
    
    # Now you can use args.seed to set your seed
    seed = args.seed
    iteration = args.index
    state_dir = f'state_files/seed_{seed}'

    
    initial, scenario = load_state(state_dir, iteration, seed)
    
    if initial is None:
        # Initial setup if no saved state exists
        print('initialized')
        model = gpt2(seed)
        scenario = Scenario(model.configspace, deterministic=False, output_directory=f'state_files/seed_{seed}', n_trials=100, seed=seed)
        initial = RandomInitialDesign(scenario, n_configs=5)
        intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=1)
        smac = HyperparameterOptimizationFacade(scenario, model.train, intensifier=intensifier, initial_design=initial, overwrite=True)
    else:
        model = gpt2(seed)
        n_configs = max(0,6-iteration)
        print(iteration, n_configs)
        initial = RandomInitialDesign(scenario, n_configs=n_configs)
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
                scenario,
                max_config_calls=1
            )
        smac = HyperparameterOptimizationFacade(
            scenario,
            model.train,
            intensifier=intensifier,
            initial_design=initial,
            overwrite=False
        )

    info = smac.ask()
    cost = model.train(config=info.config, seed=info.seed)
    value = TrialValue(cost=cost, time=0.5)
    smac.tell(info, value)
    
    smac.scenario.save()
    save_state(initial, state_dir, iteration)
    
    """
    # Save the state and exit
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, f'initial_state_{iteration}.pkl'), 'wb') as f:
        pickle.dump(initial, f)
    
    iteration += 1
    print('==========',iteration)
    initial_path = os.path.join(state_dir, f'initial_state_{iteration-1}.pkl')
    with open(initial_path, 'rb') as f:
        reloaded_initial = pickle.load(f)
    print('reloaded initial')
    reloaded_scenario = Scenario.load(find_newest_directory(state_dir)/f'{seed}')
    print('reloaded scenario')
    
    reloaded_intensifier = HyperparameterOptimizationFacade.get_intensifier(
            reloaded_scenario,
            max_config_calls=1
        )
    reloaded_smac = HyperparameterOptimizationFacade(
        reloaded_scenario,
        model.train,
        intensifier=reloaded_intensifier,
        initial_design=reloaded_initial,
        overwrite=False
    )
    if reloaded_smac is not None and reloaded_scenario is not None:
        verify_loaded_state(smac, reloaded_smac, scenario, reloaded_scenario)
    else:
        print("Error: Could not reload saved state for verification.")
    

    info = reloaded_smac.ask()
    cost = model.train(config=info.config, seed=info.seed)
    value = TrialValue(cost=cost, time=0.5)
    reloaded_smac.tell(info, value)
    
    reloaded_smac.scenario.save()
    save_state(initial, state_dir, iteration)
    
    iteration += 1
    
    print('==========',iteration)
    initial_path = os.path.join(state_dir, f'initial_state_{iteration-1}.pkl')
    with open(initial_path, 'rb') as f:
        reloaded_initial1 = pickle.load(f)
    print('reloaded initial')
    reloaded_scenario1 = Scenario.load(find_newest_directory(state_dir)/f'{seed}')
    print('reloaded scenario')
    
    reloaded_intensifier1 = HyperparameterOptimizationFacade.get_intensifier(
            reloaded_scenario1,
            max_config_calls=1
        )
    reloaded_smac1 = HyperparameterOptimizationFacade(
        reloaded_scenario1,
        model.train,
        intensifier=reloaded_intensifier1,
        initial_design=reloaded_initial1,
        overwrite=False
    )
    if reloaded_smac1 is not None and reloaded_scenario1 is not None:
        verify_loaded_state(reloaded_smac, reloaded_smac1, reloaded_scenario, reloaded_scenario1)
    else:
        print("Error: Could not reload saved state for verification.")

    
    
    model = gpt2(seed)
    
    state_dir = f'state_files/seed_{seed}'
    if index > 1 and find_newest_directory(state_dir) is not None:
        print('is not None')
        saved_scenario = Scenario.load(find_newest_directory(state_dir)/f'{seed}')
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1
        )
        smac = HyperparameterOptimizationFacade(
            saved_scenario,
            model.train,
            intensifier=intensifier,
            overwrite=False
        )
        print('set up: smac loaded previous',index)
    else:
        # Initialize SMAC for the first time
        scenario = Scenario(model.configspace, deterministic=False, output_directory=f'state_files/seed_{seed}', n_trials=100, seed=seed)
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

    model1 = gpt2(seed)
    
    assert model1.configspace==model.configspace
    print('model again')
    state_dir = f'state_files/seed_{seed}'
    saved_scenario = Scenario.load(find_newest_directory(state_dir)/f'{seed}')
    print('saved_scenario again')
    intensifier1 = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1
        )
    print('intensifier')
    smac1 = HyperparameterOptimizationFacade(
            saved_scenario,
            model1.train,
            intensifier=intensifier1,
            initial_design=
            overwrite=False
        )
    print('smac')
    print(smac.runhistory.finished,smac1.runhistory.finished)
    """
