import statistics
import collections
from typing import Tuple

import numpy as np
import gym
import torch
from torch import nn, optim, Tensor, tensor
import torch.nn.functional as F
import tqdm

EPS = np.finfo(np.float32).eps.item()
DISCRETISATION_SIZE = 100

# Create the environment
env = gym.make("InvertedPendulum-v2")

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
torch.manual_seed(seed)

class ActorCritic(nn.Module):
    def __init__(
        self, 
        state_size: int,
        num_hidden_units: int
    ) -> None:
        super().__init__()

        self.common = nn.Linear(
            in_features=state_size,
            out_features=num_hidden_units
        )
        self.actor = nn.Linear(
            in_features=num_hidden_units,
            out_features=DISCRETISATION_SIZE
        )
        self.critic = nn.Linear(
            in_features = num_hidden_units,
            out_features=1
        )

    def forward(self, inputs: Tensor):
        x = F.relu(self.common(inputs))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_prob, state_values

state_size = env.observation_space.shape[0]
num_hidden_units = 128

model = ActorCritic(state_size, num_hidden_units)

def env_step(env, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  state, reward, done, _ = env.step([action])
  return state, reward, done

def run_episode(
    env: gym.Wrapper,
    model: nn.Module,
    max_steps: int
) -> Tuple[Tensor, Tensor, Tensor]:
    saved_action_probs = torch.zeros(max_steps, dtype=torch.float32)
    saved_state_values = torch.zeros(max_steps, dtype=torch.float32)
    saved_rewards = torch.zeros(max_steps, dtype=torch.float32)

    initial_state = env.reset()
    state = initial_state

    for i in range(max_steps):
        state = torch.from_numpy(state).float()
        action_probs, value = model(state)
        
        saved_state_values[i] = value
        
        action = torch.distributions.Categorical(probs=action_probs).sample()
        saved_action_probs[i] = action_probs[action]

        state, reward, done = env_step(env, np.linspace(-3,3,DISCRETISATION_SIZE)[action.item()])
        saved_rewards[i] = reward

        if bool(done):
            break

    return saved_action_probs[:i+1], saved_state_values[:i+1], saved_rewards[:i+1]

def get_expected_returns(
    rewards: Tensor,
    gamma: float,
    standardise: bool = True,
) -> Tensor:
    n = len(rewards)
    discounted_sum = 0.0
    returns = torch.zeros_like(rewards)

    for i in range(n):
        reward = rewards[n-i-1]
        discounted_sum = reward + gamma*discounted_sum
        returns[n-i-1] = discounted_sum
        pass

    if standardise:
        returns = (returns - returns.mean())/(returns.std()+EPS)
    
    return returns

def compute_loss(
    action_probs: Tensor,
    values: Tensor,
    returns: Tensor,
) -> Tensor:
    advantages = returns - values

    action_log_probs = torch.log(action_probs)
    action_loss = -(action_log_probs*advantages).sum()

    critic_loss = F.huber_loss(values, returns, reduction='sum')

    return action_loss + critic_loss

def train_step(
    env: gym.Wrapper,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    gamma: float,
    max_steps_per_episode: int
) -> Tensor:
    model.train()

    optimiser.zero_grad()
    action_probs, values, rewards = run_episode(
        env, model, max_steps_per_episode
    )
    returns = get_expected_returns(rewards, gamma)
    loss = compute_loss(action_probs, values, returns)
    loss.backward()
    optimiser.step()

    return rewards.sum()

optimizer = optim.Adam(model.parameters(), lr=0.001)

min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.95

episodes_reward = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
    for i in t:
        episode_reward = int(
            train_step(env, model, optimizer, gamma, max_steps_per_episode)
        )

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        # env.render()

        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show average episode reward every 10 episodes
        if i % 10 == 0:
            # print(f'Episode {i}: average reward: {running_reward}')
            pass

        if running_reward > reward_threshold and i >= min_episodes_criterion:  
            break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
