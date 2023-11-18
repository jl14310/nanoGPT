from ConfigSpace import Constant, Configuration, ConfigurationSpace, Float,Integer, Categorical, Normal

from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import RandomInitialDesign
from smac.runhistory.dataclasses import TrialValue
import argparse
import subprocess
import json
import os 
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-2 training with specified seed.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Now you can use args.seed to set your seed
    seed = args.seed

    model = gpt2(seed)
    
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
    for i in range(30):
        print(i)
        info = smac.ask()
        assert info.seed is not None
        print(i, 'info')
        cost = model.train(config=info.config,seed=info.seed)
        print(i, 'cost')
        value = TrialValue(cost=cost, time=0.5)
        print(i, 'value')
        smac.tell(info, value)
