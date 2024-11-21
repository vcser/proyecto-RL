import matplotlib.pyplot as plt
import time
import cv2
import gymnasium as gym
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from collections import deque
import random
import ale_py

gym.register_envs(ale_py)

# Crear el entorno de Pacman
env = gym.make("ALE/MsPacman-v5", render_mode='rgb_array', frameskip=4)


def preprocess_observation(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=0) / 255.0


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
        )
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        return self.policy(shared_out), self.value(shared_out)


class PPOAgent:
    def __init__(self, input_shape, num_actions, learning_rate=1e-4, gamma=0.99, epsilon=0.2, entropy_beta=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_beta = entropy_beta
        self.policy_net = PolicyNetwork(input_shape, num_actions)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def train(self, states, actions, old_log_probs, returns, advantages):
        states = torch.stack(states)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(old_log_probs)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)

        for _ in range(4):
            logits, values = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - old_log_probs)
            clipped_ratios = torch.clamp(
                ratios, 1 - self.epsilon, 1 + self.epsilon)
            surrogate_loss = torch.min(
                ratios * advantages, clipped_ratios * advantages).mean()

            value_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = -surrogate_loss + 0.5 * value_loss - self.entropy_beta * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Policy Loss: {-surrogate_loss.item()}, Value Loss: {
                  value_loss.item()}, Entropy: {entropy.item()}")



def evaluate(env, agent, num_episodes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.policy_net.to(device)
    agent.policy_net.eval()
    
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(preprocess_observation(state), dtype=torch.float32).to(device)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                logits, _ = agent.policy_net(state.unsqueeze(0))
                action = torch.argmax(logits).item()
            next_state, reward, done, _, _ = env.step(action)
            state = torch.tensor(preprocess_observation(next_state), dtype=torch.float32).to(device)
            total_reward += reward

        total_rewards.append(total_reward)

    mean_reward = np.mean(total_rewards)
    print(f"Evaluation - Average Reward: {mean_reward}")
    return mean_reward


# Inicializa una lista para guardar las recompensas
reward_history = []


def train_ppo(env, agent, num_episodes=5000, rollout_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.policy_net.to(device)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(preprocess_observation(
            state), dtype=torch.float32).to(device)
        done = False
        total_reward = 0

        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        while not done:
            for _ in range(rollout_length):
                logits, value = agent.policy_net(state.unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

                next_state, reward, done, _, _ = env.step(action.item())
                next_state = torch.tensor(preprocess_observation(
                    next_state), dtype=torch.float32).to(device)

                log_prob = dist.log_prob(action).item()
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob)
                values.append(value.squeeze().item())

                total_reward += reward
                state = next_state

                if done:
                    break

            _, next_value = agent.policy_net(state.unsqueeze(0))
            returns = agent.compute_returns(
                rewards, dones, values, next_value.item())
            advantages = returns - torch.tensor(values)

            agent.train(states, actions, log_probs, returns, advantages)
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        reward_history.append(total_reward)
        print(f"Episode {episode} - Total Reward: {total_reward}")

        # Graficar recompensa acumulada cada 100 episodios
        if episode % 100 == 0:
            evaluate(env, agent, num_episodes=10)
            plt.plot(reward_history)
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.show()


def play(env, agent, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.policy_net.to(device)
    agent.policy_net.eval()  # Cambiar al modo de evaluación

    state, _ = env.reset()
    state = torch.tensor(preprocess_observation(
        state), dtype=torch.float32).to(device)
    total_reward = 0
    done = False

    while not done:
        # Obtener la acción del modelo
        with torch.no_grad():
            logits, _ = agent.policy_net(state.unsqueeze(0))
            action = torch.distributions.Categorical(
                logits=logits).sample().item()

        # Ejecutar la acción en el entorno
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(preprocess_observation(
            next_state), dtype=torch.float32).to(device)

        # Renderizar el entorno
        if render:
            frame = env.render()
            cv2.imshow("Pacman", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(10) & 0xFF == ord('q'):  # Presionar 'q' para salir
                break

        total_reward += reward
        state = next_state
        time.sleep(0.01)  # Para ralentizar el renderizado y hacerlo visible

    env.close()
    cv2.destroyAllWindows()
    print(f"Total Reward: {total_reward}")


def random_agent(env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)

    mean_reward = np.mean(total_rewards)
    print(f"Random Agent - Average Reward: {mean_reward}")
    return mean_reward



# Crear el agente y entrenarlo
input_shape = (1, 84, 84)
num_actions = env.action_space.n
ppo_agent = PPOAgent(input_shape, num_actions)

random_reward = random_agent(env)
print(f"Trained Agent Average Reward: {evaluate(env, ppo_agent)}")

train_ppo(env, ppo_agent, num_episodes=500)
torch.save(ppo_agent.policy_net.state_dict(), "ppo_pacman_trained.pth")

# ppo_agent.policy_net.load_state_dict(torch.load("ppo_pacman_trained.pth"))  # Carga el modelo entrenado
play(env, ppo_agent, render=True)
