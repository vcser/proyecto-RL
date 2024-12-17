import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import cv2
import torch.distributions as dist
from torch import nn


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3)
        self.convolution2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5)
        self.convolution3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=7)
        self.fc1 = nn.Linear(in_features=1792, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=9)

    def forward(self, x):
        # x = x.cuda()
        x.reshape(-1, 3, 210, 160)
        x = F.relu(F.max_pool2d(self.convolution1(x), 3))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.reshape(x.size(0), - 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def preprocess_observation(state):
    return np.reshape(state, (1, 210, 160, 3)).transpose(0, 3, 1, 2)/255


def select_action(state, policy_net):
    with torch.no_grad():
        # Usar la red neuronal para seleccionar la acción
        state_tensor = state.float()
        action = policy_net(state_tensor).argmax().item()
    return action


def play_game(env, policy_net):
    state = env.reset()
    state = torch.tensor(preprocess_observation(state), dtype=torch.float32)
    total_reward = 0
    done = False

    while not done:
        # Seleccionar la acción basada en la política aprendida
        action = select_action(state, policy_net)

        # Tomar la acción en el entorno
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Actualizar el estado
        state = torch.tensor(preprocess_observation(
            next_state), dtype=torch.float32)

        # Renderizar el entorno para ver cómo juega el agente
        try:
            env.render()
        except:
            pass
        # env.render(mode="human")

    print(f"Total Reward: {total_reward}")


if __name__ == "__main__":

    # Configurar el entorno
    env = gym.make('MsPacman-v4', render_mode='human')
    env.metadata['render_fps'] = 30

    # Crear la red neuronal y cargar los pesos entrenados
    # Ajusta los parámetros si es necesario
    policy_net = CNN()
    policy_net.load_state_dict(torch.load("pacman_dqn_episode_23.pth"))
    policy_net.eval()

    play_game(env, policy_net)

    # Ejecutar el juego

    # Cerrar el entorno después de jugar
    env.close()
