from ConfigSpace import Constant, Configuration, ConfigurationSpace, Float,Integer, Categorical, Normal
import os
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue
import numpy as np
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
import subprocess
import json
class resnet:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed = 0)
        batch_size = Integer('batch_size',(8,13),default = 12)
        block_size = Constant('block_size',1024)#Integer('block_size',(1024,1024),default = 1024)
        learning_rate = Float('learning_rate',(1e-6,1e-3),default = 6e-4)
        max_iters = Constant('max_iters',30)#Integer('max_iters',(5000,5000),default = 5000)
        weight_decay = Float('weight_decay',(1e-3,1e0),default = 1e-1)
        lr_decay_iters = Constant('lr_decay_iters',30)
        cs.add_hyperparameters([batch_size,block_size,learning_rate,max_iters,lr_decay_iters,weight_decay])
        return cs


    def train(self,config:Configuration,seed:int=0):
    # Convert the hyperparameters to their appropriate types
        batch_size = int(config['batch_size'])
        block_size = int(config['block_size'])
        learning_rate = config['learning_rate']
        max_iters = int(config['max_iters'])
        lr_decay_iters = int(config['lr_decay_iters'])
        weight_decay = config['weight_decay']

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
        }

        results = ''.join(['{name} = {value}\n'.format(name=k, value=v) for k, v in yaml_config.items()])
        print(results)
        with open('gpt2_config.py', 'w') as f:
            f.write(results)
        command = ['python', 'train.py', 'gpt2_config.py']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()

        # Read the evaluation score from the subprocess output
        results = None
        with open('results.json') as f:
            results = json.load(f)
            # print(results)
        evaluation_score = float(results['best_val_loss'][-1])
        print(evaluation_score)

        return evaluation_score 





if __name__ == "__main__":
    model = resnet()
    print('model')
    # Scenario object
    scenario = Scenario(model.configspace, deterministic=False, n_trials=100)
    print('scenario')
    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,  # We basically use one seed per config only
    )
    print('intensifier')
    os.environ['WANDB_MODE'] = 'offline'
    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,
        intensifier=intensifier,
        overwrite=True,
    )
    print('smac')
    
    # We can ask SMAC which trials should be evaluated next
    for i in range(10):
        print(i)
        info = smac.ask()
        assert info.seed is not None
        print('info')
        
        cost = model.train(config=info.config, seed=info.seed)
        print('cost')
        value = TrialValue(cost=cost, time=0.5)
        print('value')
        smac.tell(info, value)
