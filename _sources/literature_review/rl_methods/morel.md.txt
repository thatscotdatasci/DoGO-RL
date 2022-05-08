# MoREL: Model-Based Offline Reinforcement Learning

- Paper link: [Arxiv](https://arxiv.org/abs/2005.05951)

## Key Points

The following were the key takeaways:

- Learn a pessimistic MDP (P-MDP) and then learn a near-optimal policy in this P-MDP
- _"Performance in the real environment is approximately lower-bounded by the performance in the P-MDP"_
- Use data collected by one of more data logging (behaviour) policies
- _"Pessimistic MDP penalises policies that venture into unknown parts of state-action space."_
- HALT absorbing state used to heavily punish policies that visit unknown states. This is an absorbing state, and so effectively ends the episode.
- USAD uses an ensemble of MLP models to quantify uncertainty. This is likely to be an underestimation due to the overconfidence issues of such networks, and are at significant risk of suffering from adversarial examples.
- _"MORel incorporates both generalisation and pessimism. This enables MOReL to perform policy improvement in known states that may not directly occur in the static offline dataset, but nevertheless can be predicted using the dataset by leveraging the power of generalisation."_
- Authors did not find the magnitude of the reward penalty for entering the HALT state to be important, as long as it was $\le$ 0. However, this was for simple environments; authors uncertain what impact it may have in other environments.
