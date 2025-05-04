import cv2
import gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class SimpleDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(SimpleDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_out_size = self._get_conv_output_size(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_output_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return self.fc(conv_out)
    
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Agent(object):
    def __init__(self):
        set_random_seed()
        
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cpu")
        
        state_shape = (4, 84, 84)
        self.model = SimpleDQN(state_shape, self.action_space.n).to(self.device)
        
        self.load_model("best_weight.pth")
        
        self.frame_buffer = deque(maxlen=4)
        self.is_initialized = False
        
        self.skip_frame_counter = 0
        self.skip_frames = 4
        self.last_action = 0
        
    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['q_network'])
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def act(self, observation):
        processed_frame = self.preprocess_frame(observation)
        
        if not self.is_initialized:
            for _ in range(4):
                self.frame_buffer.append(processed_frame)
            
            self.is_initialized = True
            self.skip_frame_counter = 0
        else:
            if self.skip_frame_counter == 0:
                self.frame_buffer.append(processed_frame)
                
        self.skip_frame_counter = (self.skip_frame_counter + 1) % self.skip_frames
        
        if self.skip_frame_counter == 1:
            state = np.array(self.frame_buffer)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                self.last_action = q_values.max(1)[1].item()
        
        return self.last_action