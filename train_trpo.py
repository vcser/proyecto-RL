import cv2
import gymnasium as gym
import torch
import numpy as np
from torch import nn
import torch.distributions as dist
import ale_py

gym.register_envs(ale_py)

from torch.optim import Adam

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x)
    
class ValueNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.network(x)

def preprocess_observation(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=0) / 255.0

def kl_divergence(old_action_probs, new_action_probs):
    return (old_action_probs * (torch.log(old_action_probs) - torch.log(new_action_probs))).sum(-1).mean()

def compute_kl_divergence(old_policy, new_policy, states, actions):
    """Compute KL divergence between the old and new policies."""
    with torch.no_grad():
        old_log_probs = dist.Categorical(logits=old_policy(states)).log_prob(actions)
        new_log_probs = dist.Categorical(logits=new_policy(states)).log_prob(actions)
        kl_div = (old_log_probs - new_log_probs).mean()
    return kl_div

# Function for line search
def line_search(policy_net, old_policy_net, loss_fn, states, actions, advantages, max_kl, step_size=1.0, beta=0.5, max_attempts=10):
    """Perform line search to enforce KL constraint."""
    # Save the old policy's parameters
    old_policy_params = {k: v.clone() for k, v in old_policy_net.state_dict().items()}

    for attempt in range(max_attempts):
        # Try updating the policy with the current step size
        for param, old_param in zip(policy_net.parameters(), old_policy_params.values()):
            param.data.copy_(old_param + step_size * (param - old_param))

        # Compute loss and KL divergence
        kl_div = compute_kl_divergence(old_policy_net, policy_net, states, actions)
        policy_loss = loss_fn(policy_net(states), actions, advantages)

        # Check if KL divergence constraint is satisfied
        if kl_div <= max_kl:
            print(f"Line search successful on attempt {attempt + 1}, KL divergence: {kl_div:.6f}")
            return policy_loss

        # If not, reduce step size and retry
        step_size *= beta
        print(f"Line search reducing step size to {step_size:.6f} (KL: {kl_div:.6f})")

    # If all attempts fail, revert to the old policy
    print("Line search failed to satisfy KL constraint; reverting policy update.")
    policy_net.load_state_dict(old_policy_net.state_dict())
    return None

# Define loss function (e.g., policy gradient loss)
def loss_fn(logits, actions, advantages):
    log_probs = logits.log_softmax(dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    return -(selected_log_probs * advantages).mean()

def train(env, input_shape, policy_net, value_net, num_episodes, gamma=0.95, lam=0.97, max_kl=0.01):
    optimizer = Adam(value_net.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    value_net.to(device)

    for episode in range(num_episodes):
        # Step 1: Collect trajectories
        states, actions, rewards, dones = [], [], [], []
        state, _ = env.reset()
        state = torch.tensor(preprocess_observation(state), dtype=torch.float32).to(device)
        episode_reward = 0

        while True:
            # Get action from policy network
            # Sample action for a discrete action space
            logits = policy_net(state.unsqueeze(0))
            action_dist = dist.Categorical(logits=logits)  # Convert logits to categorical distribution
            action = action_dist.sample()
            actions.append(action.item())  # Append sampled action (scalar value)

            # Execute action in the environment
            next_state, reward, done, _, _ = env.step(int(action.item()))
            states.append(state)
            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            state = torch.tensor(preprocess_observation(next_state), dtype=torch.float32).to(device)
            if done:
                break

        # Step 2: Compute returns and advantages
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + gamma * G * (1 - done)
            returns.insert(0, G)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        baseline = value_net(torch.tensor(np.array(states), dtype=torch.float32).to(device)).squeeze()
        advs = returns - baseline.detach()

        # Step 3: Update policy (stub; implement line search and KL constraint manually or use an optimization library)
        old_policy_net = PolicyNetwork(input_shape, env.action_space.n).to(device)
        old_policy_net.load_state_dict(policy_net.state_dict())

        # Perform line search for policy update
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        advantages = torch.tensor(advs).to(device)

        new_loss = line_search(policy_net, old_policy_net, loss_fn, states, actions, advantages, max_kl=0.01)
        if new_loss is None:
            print("Policy update skipped due to KL divergence constraint.")

        # Step 4: Update value network
        value_loss = nn.MSELoss()(baseline, returns)
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward}")

        # Optional: Save models periodically
        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"policy_net_trpo_episode_{episode + 1}.pth")
            torch.save(value_net.state_dict(), f"value_net_trpo_episode_{episode + 1}.pth")

    # Save final models
    torch.save(policy_net.state_dict(), "policy_net_trpo_final.pth")
    torch.save(value_net.state_dict(), "value_net_trpo_final.pth")

if __name__ == "__main__":
    # Crear el entorno de Pacman
    env = gym.make("ALE/MsPacman-v5", frameskip=4)

    # Initialize policy and value networks
    input_shape = (1, 84, 84)
    policy_net = PolicyNetwork(input_shape, env.action_space.n)
    value_net = ValueNetwork(input_shape)

    # Train TRPO
    train(env, input_shape, policy_net, value_net, num_episodes=5000)

    # Close the environment
    env.close()