# Causal Confusion in Imitation Learning

- Paper link: [Arxiv](https://arxiv.org/abs/1905.11979)

## Key Points

- Treating policy learning as a supervised learning task can lead to causal misidentification due to distributional shift
- Characterised by generalisation performance worsening as more information becomes available (good brake light example)
- Causal misidentification := phenomenon whereby cloned policies fail by misidentifying the causes of expert actions
- Proposes two methods of resolving: _expert query mode_ and _policy execution mode_
- Whilst they work well, both methods require interaction with the environment via the actions. In the former we query the expert based on stats with high disagreement, in the latter we collect rewards based on executing $\pi_G$
- need to read up more on _FCMs: Functional Causal Models_
- Clear to me how this could be an issue for offline RL; unclear how we could address
- Possible to consider FCMs in advance? Could hold-out data, but probably don't have enough.
- Training with dropout/ ensembling is interesting
