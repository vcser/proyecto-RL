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


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x)


BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 100000  # Cambiado para explorar más
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000  # Ajuste para actualizaciones más frecuentes en pasos
REPLAY_MEMORY_SIZE = 10000


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
 
    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def preprocess_observation(obs):
    resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=0) / 255.0


def train(env, policy_net, target_net, memory, optimizer, num_episodes):
    epsilon = EPSILON_START
    steps_done = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net.to(device)
    target_net.to(device)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(preprocess_observation(state), dtype=torch.float32).to(device)
        total_reward = 0

        while True:
            # Selección de acción con epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = state.unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()

            # Ejecutar la acción en el entorno
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(preprocess_observation(next_state), dtype=torch.float32).to(device)
            total_reward += reward

            # Guardar la transición en el buffer de repetición
            memory.push((state.cpu(), action, reward, next_state.cpu(), done))
            state = next_state

            # Actualizar el valor de epsilon para exploración-explotación
            epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
            steps_done += 1

            # Entrenar la red neuronal cada cierto número de pasos
            if steps_done % 4 == 0 and len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))

                # Convertir las transiciones a tensores en la GPU
                state_batch = torch.stack(batch[0]).to(device)
                action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(device)
                reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
                next_state_batch = torch.stack(batch[3]).to(device)
                done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

                # Implementación de Double DQN
                q_values = policy_net(state_batch).gather(1, action_batch)
                next_actions = policy_net(next_state_batch).argmax(1, keepdim=True)
                next_q_values = target_net(next_state_batch).gather(1, next_actions).detach()
                expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

                # Calcular la pérdida y actualizar la red
                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1)  # Gradient clipping
                optimizer.step()

            # Actualización suave de la red de destino
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Guardar el modelo periódicamente
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), f"pacman_dqn_episode_{episode}.pth")

        print(f"Episode {episode} - Total Reward: {total_reward}")

    # Guardar el modelo final después de todos los episodios
    torch.save(policy_net.state_dict(), "pacman_dqn_final.pth")


policy_net = DQN((3, 84, 84), env.action_space.n)
policy_net.load_state_dict(torch.load("pacman_dqn_final.pth"))
policy_net.eval()
target_net = DQN((3, 84, 84), env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

train(env, policy_net, target_net, memory, optimizer, num_episodes=5000)
