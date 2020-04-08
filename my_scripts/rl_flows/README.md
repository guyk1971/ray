# rl_flows
the files in this folder are scripts that implement various rl flows:
(should be equivalent to the `train` folder in a project structure)
1. training agent on either env or buffer (batch) reading yaml config file
1. rollout agent on env and saving buffer
1. evaluating an agent on a provided buffer (the output is csv with predictions)

We need to support the following:
1. Train new agent on simulator &rarr; model parameters
1. run (loaded/new) agent on simulator &rarr; experience buffer
1. Train agent on experience buffer &rarr; model parameters
1. go to 2

to support that we'll have `.py` files and `.yaml` for configuration

# Scripts
There will be at least 2 types of scripts:
1. A template for `train_<agent>.py` - this will serve as a template from which we'll create a script per agent. 
The reason is that each agent has its own internal flow and variables that we might want to track while training 
so the callbacks might be different. we'll start with the following:
    1. `train_dqn.py` 
    2. `train_dbcq.py`
    3. `train_ppo.py`  
The main role of these scripts will be to train an agent and optionally record debug info to track 
training progress. 
the various modes of training will be supported via `yaml` config files:  
    - train new agent or load pre-trained to continue training
    - train on simulator or static experience buffer (while evaluating on sim or OPE)
    - can optionally output experience buffer to file 
    
    
1. `run_agent.py` - this will run an agent on simulator env without training and will dump 
the experience to file. 

In addition, we might want to test the policy on experience buffer that we're provided to test the performance against 
oracle. (although this doesnt make too much sense.why not run on the env ?)  
we might also want to perform OPE on a static buffer without training. 


# Yaml files

## Environments
if you want to know which gym environments are registered:  
```
gym.env.registry.env_specs
```