import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
import gym_super_mario_bros

from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack


class FrameSkipper(gym.Wrapper):
    def __init__(self, environment, skip_count=4):
        super().__init__(environment)
        self._skip_count = skip_count

    def step(self, action):
        accumulated_reward = 0.0
        terminal = False
        metadata = {}
        for _ in range(self._skip_count):
            observation, reward, terminal, metadata = self.env.step(action)
            accumulated_reward += reward
            if terminal:
                break
        return observation, accumulated_reward, terminal, metadata


class InitialRandomActions(gym.Wrapper):
    def __init__(self, environment, max_random_actions=30):
        gym.Wrapper.__init__(self, environment)
        self.max_random_count = max_random_actions
        self.override_random_count = None
        self.no_action = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_random_count is not None:
            action_count = self.override_random_count
        else:
            action_count = self.unwrapped.np_random.randint(1, self.max_random_count + 1)
        assert action_count > 0
        result = None
        for _ in range(action_count):
            result, _, terminated, _ = self.env.step(self.no_action)
            if terminated:
                result = self.env.reset(**kwargs)
        return result

    def step(self, action):
        return self.env.step(action)
    

class ImageProcessor(gym.ObservationWrapper):
    def __init__(self, environment):
        super().__init__(environment)
        
    def observation(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        return resized_frame


def prepare_environment():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = FrameSkipper(env)
    env = InitialRandomActions(env, max_random_actions=8)
    env = ImageProcessor(env)
    env = FrameStack(env, 4)
    return env


class DeepQNetwork(nn.Module):
    def __init__(self, input_dimensions, output_size):
        super(DeepQNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dimensions[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_output_size = self._calculate_conv_output_size(input_dimensions)
        
        self.decision_maker = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        
    def _calculate_conv_output_size(self, dimensions):
        dummy_input = torch.zeros(1, *dimensions)
        output = self.feature_extractor(dummy_input)
        return int(np.prod(output.size()))
    
    def forward(self, state):
        features = self.feature_extractor(state)
        return self.decision_maker(features)


class ExperienceMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)
    
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def retrieve_batch(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def size(self):
        return len(self.memory)


class MarioAgent:
    def __init__(self, state_shape, action_space, params):
        self.state_shape = state_shape
        self.action_space = action_space
        self.action_count = action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.discount_factor = params['discount_factor']
        self.explore_start = params['explore_start']
        self.explore_min = params['explore_min']
        self.explore_decay = params['explore_decay']
        self.sync_networks_every = params['sync_networks_every']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        
        self.memory = ExperienceMemory(params['memory_capacity'])
        
        self.policy_network = DeepQNetwork(state_shape, self.action_count).to(self.device)
        self.target_network = DeepQNetwork(state_shape, self.action_count).to(self.device)
        self.sync_target_with_policy()
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        self.training_step = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def choose_action(self, state, evaluation_mode=False):
        if not evaluation_mode:
            exploration_rate = self.explore_min + (self.explore_start - self.explore_min) * \
                    np.exp(-self.training_step / self.explore_decay)
        else:
            exploration_rate = 0.01
        
        if random.random() < exploration_rate:
            return random.randrange(self.action_count)
        
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array([state])).to(self.device)
                q_values = self.policy_network(state_tensor)
                return q_values.max(1)[1].item()
        
    def learn(self):
        if self.memory.size() < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.retrieve_batch(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.training_step % self.sync_networks_every == 0:
            self.sync_target_with_policy()
    
    def sync_target_with_policy(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def save_checkpoint(self, path):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
        return path
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']


def run_training(params, max_episodes=10000, starting_episode=1):
    env = prepare_environment()
    
    state_dimensions = (4, 84, 84)
    agent = MarioAgent(state_dimensions, env.action_space, params)
    
    if starting_episode > 1:
        checkpoint_file = f"saved_models/mario_ep{starting_episode}.pth"
        agent.load_checkpoint(checkpoint_file)
        print(f"Resumed training from {checkpoint_file}")
    
    os.makedirs("saved_models", exist_ok=True)
    
    step_counter = 0
    learning_begins = params['learning_begins']
    
    for episode in range(starting_episode, max_episodes + 1):
        current_state = env.reset()
        episode_done = False
        episode_step_count = 0
        episode_reward = 0
        
        while not episode_done:
            chosen_action = agent.choose_action(current_state)
            next_state, reward, episode_done, info = env.step(chosen_action)
            
            agent.memory.add_experience(current_state, chosen_action, reward, next_state, episode_done)
            
            current_state = next_state
            step_counter += 1
            episode_step_count += 1
            agent.training_step += 1
            episode_reward += reward
            
            if step_counter > learning_begins and step_counter % 2 == 0:
                agent.learn()
        
        print(f"Episode {episode}: Reward = {episode_reward}, Steps = {episode_step_count}")
        
        if episode % 500 == 0:
            checkpoint_file = f"saved_models/mario_ep{episode}.pth"
            agent.save_checkpoint(checkpoint_file)
            print(f"Progress saved to {checkpoint_file}")
    
    env.close()


if __name__ == "__main__":
    training_params = {
        'discount_factor': 0.99,
        'learning_rate': 0.0001,
        'explore_start': 0.95,
        'explore_min': 0.01,
        'explore_decay': 1000000,
        'memory_capacity': 100000,
        'batch_size': 64,
        'sync_networks_every': 2500,
        'learning_begins': 20000,
    }
    
    run_training(training_params, max_episodes=100000)