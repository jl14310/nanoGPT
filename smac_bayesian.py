from ConfigSpace import Constant, Configuration, ConfigurationSpace, Float,Integer, Categorical, Normal

from smac import HyperparameterOptimizationFacade, Scenario, Callback
from smac.initial_design import RandomInitialDesign, SobolInitialDesign
from smac.runhistory.dataclasses import TrialValue
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue

from smac.runhistory.runhistory import RunHistory
from smac.intensifier.intensifier import Intensifier

import argparse
import subprocess
import json
import os 
import pickle 

class gpt2:
    def __init__(self, seed, modeltype):
        self.seed = seed
        self.modeltype = modeltype
        
    @property
    def configspace(self) -> ConfigurationSpace:
        if self.modeltype == 'nano':
            cs = ConfigurationSpace(seed = self.seed)
            n_layer = Constant('n_layer',6)
            n_head = Constant('n_head',6)
            n_embd = Constant('n_embd',384)
            dropout = Constant('dropout',0.2)
            gradient_accumulation_steps = Constant('gradient_accumulation_steps',1)
            
            batch_size = Integer('batch_size',(50,100),default = 64)
            block_size = Constant('block_size',256)
            learning_rate = Float('learning_rate',(1e-6,1e-3),default = 1e-3)
            max_iters = Constant('max_iters',10000)
            weight_decay = Float('weight_decay',(1e-3,1e-1),default = 1e-1)
            lr_decay_iters = Constant('lr_decay_iters',10000)
            warmup_iters = Constant('warmup_iters',100)
            seed = Constant('seed',self.seed)
            min_lr = Constant('min_lr',1e-4)
            cs.add_hyperparameters([n_layer,n_head,n_embd,dropout,gradient_accumulation_steps,batch_size,block_size,learning_rate,max_iters,lr_decay_iters,warmup_iters,weight_decay,seed,min_lr])
        else:
            cs = ConfigurationSpace(seed = self.seed)
            batch_size = Integer('batch_size',(8,13),default = 12)
            block_size = Constant('block_size',1024)
            learning_rate = Float('learning_rate',(1e-6,1e-3),default = 6e-4)
            max_iters = Constant('max_iters',5000)
            weight_decay = Float('weight_decay',(1e-3,1e0),default = 1e-1)
            lr_decay_iters = Constant('lr_decay_iters',5000)
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
        
        if self.modeltype =='nano':
            n_layer = int(config['n_layer'])
            n_head = int(config['n_head'])
            n_embd = int(config['n_embd'])
            dropout = config['dropout']
            gradient_accumulation_steps = int(config['gradient_accumulation_steps'])
            warmup_iters = int(config['warmup_iters'])
            min_lr = config['min_lr']
            # Generate the YAML configuration file
            yaml_config = {
                'wandb_run_name': "'nano'",
                'n_layer':n_layer,
                'n_head':n_head,
                'n_embd':n_embd,
                'dropout':dropout,
                'gradient_accumulation_steps':gradient_accumulation_steps,
                'batch_size': batch_size,
                'block_size': block_size,
                'learning_rate': learning_rate,
                'max_iters': max_iters,
                'beta2':0.99,
                'decay_lr':True,
                'warmup_iters':warmup_iters,
                'min_lr':min_lr,
                'lr_decay_iters': lr_decay_iters,
                'eval_interval': 250,
                'eval_iters': 10,
                'log_interval': 200,
                'weight_decay': weight_decay,
                'seed':seed
            }
        else:
             # Generate the YAML configuration file
            yaml_config = {
                'wandb_run_name': "'gpt2'",
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
        with open(f'config_files/config_{self.modeltype}_{seed}.conf', 'w') as f:
            f.write(config_file_content)
        command = ['python', 'train_config.py', '-f',f'config_files/config_{self.modeltype}_{seed}.conf','-c',f'seed={seed}']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()

        # Read the evaluation score from the subprocess output
        results = None
        
        os.makedirs(os.path.dirname(f'bayesian_results/results_{self.modeltype}_{seed}.json'), exist_ok=True)
        with open(f'bayesian_results/results_{self.modeltype}_{seed}.json') as f:
            results = json.load(f)
            # print(results)
        evaluation_score = float(results['best_val_loss'][-1])
        print(evaluation_score)

        return evaluation_score 


from pathlib import Path

def find_newest_directory(base_directory):
    base_path = Path(base_directory)
    if not base_path.exists():
        return None
    directories = [d for d in base_path.iterdir() if d.is_dir()]
    if not directories:
        return None
    newest_dir = max(directories, key=lambda d: d.stat().st_ctime)
    return newest_dir
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-2 training with specified seed.')
    #parser.add_argument('--model',type=str, default='nano')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    #parser.add_argument('--index', type=int, required=True, help='Index of the SLURM array job')
    args = parser.parse_args()
    
    # Now you can use args.seed to set your seed
    seed = args.seed
    #iteration = args.index
    modeltype = 'nano' # args.model
    
    print('============',seed,type(seed),'============')
    
    state_dir = f'state_files/{modeltype}_seed_{seed}'
    model = gpt2(seed,modeltype)
    
    # Scenario object
    scenario = Scenario(model.configspace, deterministic=False, n_trials=100, seed=seed)
    print('set up: scenario')
    initial = RandomInitialDesign(scenario, n_configs=8)
    
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,  # We basically use one seed per config only
    )
    print('set up: intensifier')
    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(  
        scenario,
        model.train,
        intensifier=intensifier,
        initial_design=initial,
        overwrite=True,
    )
    print('set up: smac')
    # We can ask SMAC which trials should be evaluated next
    for i in range(32):
        print('==============',i,'==============')
        info = smac.ask()
        assert info.seed is not None
        print(i, 'info')
        cost = model.train(config=info.config,seed=info.seed)
        print(i, 'cost')
        value = TrialValue(cost=cost, time=0.5)
        print(i, 'value')
        smac.tell(info, value)
    """
    scenario_path = find_newest_directory(state_dir)
    
    if scenario_path is None:
        # Initial setup if no saved state exists
        print('initialized')
        model = gpt2(seed, modeltype)
        scenario = Scenario(model.configspace, deterministic=False, output_directory=f'state_files/{modeltype}_seed_{seed}', n_trials=100, seed=seed)
        initial = RandomInitialDesign(scenario, n_configs=2)
        intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=1)
        smac = HyperparameterOptimizationFacade(scenario, model.train, intensifier=intensifier, initial_design=initial, overwrite=True)
    else:
        model = gpt2(seed,modeltype)
        
        scenario = Scenario.load(scenario_path/f'{seed}')
        initial = RandomInitialDesign(scenario, n_configs=2)
        
        #n_configs = max(0,3-iteration)
        print('======',iteration,'======')
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

    """
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
