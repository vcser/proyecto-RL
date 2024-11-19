import torch
import gymnasium as gym
import numpy as np
import ale_py
import cv2
import torch.distributions as dist

gym.register_envs(ale_py)

# Importar la red DQN que se usó para entrenar
from dqn import DQN  # Asegúrate de tener la clase DQN en otro archivo, por ejemplo, dqn_model.py
from trpo import PolicyNetwork, ValueNetwork

def preprocess_observation(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=0) / 255.0

def select_action(state, policy_net):
    with torch.no_grad():
        # Usar la red neuronal para seleccionar la acción
        state_tensor = state.unsqueeze(0)
        action = policy_net(state_tensor).argmax().item()
    return action

def select_action_trpo(state, policy_net):
    logits = policy_net(state.unsqueeze(0))
    action_dist = dist.Categorical(logits=logits)  # Convert logits to categorical distribution
    action = action_dist.sample()
    return action

def play_game(env, policy_net):
    state, _ = env.reset()
    state = torch.tensor(preprocess_observation(state), dtype=torch.float32)
    total_reward = 0
    done = False

    while not done:
        # Seleccionar la acción basada en la política aprendida
        action = select_action(state, policy_net)

        # Tomar la acción en el entorno
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Actualizar el estado
        state = torch.tensor(preprocess_observation(next_state), dtype=torch.float32)

        # Renderizar el entorno para ver cómo juega el agente
        env.render()

    print(f"Total Reward: {total_reward}")
    
def play_game_trpo(env, policy_net):
    state, _ = env.reset()
    state = torch.tensor(preprocess_observation(state), dtype=torch.float32)
    total_reward = 0
    done = False

    while not done:
        # Seleccionar la acción basada en la política aprendida
        action = select_action_trpo(state, policy_net)

        # Tomar la acción en el entorno
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Actualizar el estado
        state = torch.tensor(preprocess_observation(next_state), dtype=torch.float32)

        # Renderizar el entorno para ver cómo juega el agente
        env.render()

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    
    # Configurar el entorno
    env = gym.make("ALE/MsPacman-v5", render_mode="human")

    use_dqn = False

    if use_dqn:
        # Crear la red neuronal y cargar los pesos entrenados
        policy_net = DQN((1, 84, 84), env.action_space.n)  # Ajusta los parámetros si es necesario
        policy_net.load_state_dict(torch.load("pacman_dqn_final.pth"))
        policy_net.eval()
        
        play_game(env, policy_net)
    else:
        policy_net = PolicyNetwork((1, 84, 84), env.action_space.n)
        policy_net.load_state_dict(torch.load("policy_net_trpo_final.pth"))
        policy_net.eval()
        
        play_game_trpo(env, policy_net)

    # Ejecutar el juego

    # Cerrar el entorno después de jugar
    env.close()
