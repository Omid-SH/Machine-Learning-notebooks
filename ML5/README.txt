
1) install package by running:

$ python setup.py develop

##############################################
##############################################

2)install other dependencies

install dependencies locally, by running:
$ pip install -r requirements.txt

##############################################
##############################################

3) code:

Blanks to be filled in are marked with "TODO"
The following files have TODOs in them:
- scripts/run_hw5_behavior_cloning.py -> ? :/
- infrastructure/rl_trainer.py -> Done
- infrastructure/torch_utils.py -> Done
- agents/bc_agent.py -> ? :/
- policies/MLP_policy.py -> Done
- policies/loaded_gaussian_policy.py -> Done

See the code + the hw pdf for more details.

##############################################
##############################################

4) run code: 

Run the following command for Imitation Learning:

$python hw5/scripts/run_hw5_behavior_cloning.py --expert_policy_file hw5/models/CartPole-v0.tar --env_name CartPole-v1 --exp_name test_bc_Cart --n_iter 1

$python hw5/scripts/run_hw5_behavior_cloning.py --expert_policy_file hw5/models/LunarLander-v2.tar --env_name LunarLander-v2 --exp_name test_bc_Lunar --n_iter 1

$python hw5/scripts/run_hw5_behavior_cloning.py --expert_policy_file hw5/models/LunarLanderContinuous-v2.tar --env_name LunarLanderContinuous-v2 --exp_name test_bc_LunarCont --n_iter 1

Run the following command for DAGGER:

$python hw5/scripts/run_hw5_behavior_cloning.py --expert_policy_file hw5/models/LunarLander-v2.tar --env_name LunarLander-v2 --exp_name test_dagger_Lunar --n_iter 10 --do_dagger

$python hw5/scripts/run_hw5_behavior_cloning.py --expert_policy_file hw5/models/LunarLanderContinuous-v2.tar --env_name LunarLanderContinuous-v2 --exp_name test_dagger_LunarCont --n_iter 10 --do_dagger

(NOTE: the --do_dagger flag, and the higher value for n_iter)

##############################################
##############################################

5) visualize saved tensorboard event file:

$ cd hw5/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)
