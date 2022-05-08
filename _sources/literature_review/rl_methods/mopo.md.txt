# MOPO: Model-based Offline Policy Optimization

- Paper link: [Arxiv](https://arxiv.org/abs/2005.13239)
- Git Repo: [GitHub](https://github.com/tianheyu927/mopo)

## Key Points

The following were the key takeaways:

- Importance for an offline RL algorithm to be able to leave the support of the training data:
  - The provided batch dataset is usually sub-optimal in terms of both the states and actions covered by the dataset
  - The target task can be different from the tasks performed in the batch data for various reasons (e.g., hard to collect for the target task)
- State-of-the-art off-policy model based (MBPO) and model-free algorithms evaluated. Model-based method and its variant without ensembles show surprisingly large gains. Suggests that model-based methods are particularly well-suited for the batch setting.
- Quantifying the risk imposed by imperfect dynamics and appropriately trading off that risk with the return is a key ingredient towards building a strong offline model-based RL algorithm
- The authors modify MBPO to to incorporate a _reward penalty_ based on an estimate of the model error. Estimate is model-dependent and does not necessarily penalise all out-of-distribution states and actions equally; prescribes penalties based on the estimated magnitude of model error.
- Estimation is done both on _states_ and _actions_, allowing generalisation to both, in contrast to model-free approaches that only reason about uncertainty with respect to actions.
- Whereas MOReL constructs terminating states based on a hard threshold on uncertainty, MOPO uses a soft reward penalty to incorporate uncertainty. This potentially allows the policy to take a few risky actions before returning to the confident area near the behavioural distribution without being terminated.
- Balance the return and risk
  - The potential gain in performance by escaping the behavioural distribution and finding a better policy
  - The risk of overfitting to the errors of the dynamics are regions far away from the behavioural distribution
- Evaluation on tasks requiring out-of-distribution generalisation

## Setting Up Their Repo

The repo has not been updated since the end of 2020. Not all library versions were fixed, and certain updates to Git protocols precented a simple installation.

The following steps were necessary:
- Clone the [MOPO Git repo](https://github.com/tianheyu927/mopo): `git clone git@github.com:tianheyu927/mopo.git`
- The following library versions were updated:
  - absl-py==0.7.0
  - cloudpickle==1.2.0
  - pyglet==1.4.0
  - tensorflow==1.14.0 (Note: the version has stayed the same, but changed away from GPU)
- Create a venv using Python 3.7.x
- Within `environments/requirements.txt`, comment out the final line: `git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl`
- Clone the [D4RL Git repo](https://github.com/rail-berkeley/d4rl) within the MOPO repo: `git clone git@github.com:rail-berkeley/d4rl.git`
- Within `d4rl/setup.py`, comment out the requirement for dm_control
- Clone the [dm_control repo](https://github.com/deepmind/dm_control): `git clone git@github.com:deepmind/dm_control.git`
- cd into `dm_control` and run `git checkout 2cb60cb9ca5921f2a82e6e371b0bcc4a1a96e610`
  - This is likely to be the version used in the paper, or at least close
- Open `dm_control/setup.py` and edit `DEFAULT_HEADERS_DIR` to wherever mujoco has been installed
- Run `pip install -e dm_control`
- Run `git config --global url."https://".insteadOf git://` - this is necessary due to changes in Git  protocols
- Run `pip install -e d4rl`
- Clone the [vikit Git repo](https://github.com/vitchyr/viskit): `git clone git@github.com:vitchyr/viskit.git`
- Run `pip install -e viskit`
- Run `pip install -e .`
