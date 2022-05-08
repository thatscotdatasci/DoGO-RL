# COMBO: Conservative Offline Model-Based Policy Optimisation

- Paper link: [Arxiv](https://arxiv.org/abs/2102.08363)

## Key Points

The following were the key takeaways:

- Does not perform direct uncertainty estimation. The authors argue that it is not necessary for offline RL. Existing model-based methods rely on some sort of strong assumption about uncertainty estimation, typically assuming access to a _model error oracle_ that can estimate upper bounds on model error for any state-action tuple. In practice, such methods use more heuristic uncertainty estimation methods, which can be difficult or unreliable for complex datasets of deep network models.
- Instead, _"COMBO trains a value function using both the offline dataset and data generated using rollouts under the model while also additionally regularising the value function on out-of-support state-action tuples generated via model rollouts. This results in a conservative estimate of the value function for out-of-support state-action tuples, without requiring explicit uncertainty estimation."_
- Effectively merges the model-free method CQL (conservative Q-learning) with the the model-free method MOPO.
- MBPO follows the standard structure of actor-critic algorithms, but in each iteration uses an augmented dataset $\mathcal{D} \cup \mathcal{D}_\text{model}$ for policy evaluation. Here, $\mathcal{D}$ is the offline dataset and $\mathcal{D}_\text{model}$ is a dataset obtained by simulating the current policy using the learned dynamics model. At each iteration, MBPO performs _k_-step rollouts using $\hat{T}$ starting from $s \in \mathcal{D}$ with a particular rollout policy $\mu(\mathbf{a}|s)$, adds the model-generated data to $\mathcal{D}_\text{model}$, and optimises the policy with a batch of data sampled from  $\mathcal{D} \cup \mathcal{D}_\text{model}$ where each datapoint in the batch is drawn from $\mathcal{D}$ with probability $f \in [0,1]$ and $\mathcal{D}_\text{model}$ with probability $1-f$.

## Current Questions/Outstanding Actions
- Proofs are unclear to me at the moment; need to read the CQL paper.
- Need to read the MBPO paper; currently unclear what the rollout policy should be. Does this need to be different from the policy being learned? If so, do we use importance sampling, or another offline technique?