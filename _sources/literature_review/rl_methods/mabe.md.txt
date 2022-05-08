# MABE

- Paper link: [Arxiv](https://arxiv.org/abs/2106.09119)

## Key Points

The following were the key takeaways:

-  Regularisation of the learned policy to keep the agent within the support of the original dataset - KL divergence with behavioural prior. Rather than using uncertainty quantification within the objective function (i.e. reward).
-  Transference of behavioural prior across various dynamics models - can train the dynamics models based on the domain, and the behavioural prior based on the task to be performed
-  Found that uncertainty quantification provided little benefit
-  Table showing other algorithms which have incorporated a behavioural prior. Apart from MOReL/MOPO, most similar is BREMEN \cite{Matsushima2020}, which is designed mainly for deployment efficient RL. Uses an unweighted prior, and small number of policy - h with implicit KL regularisation.
-  No publicly available code implementation
