import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x): return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4) # Smaller LR for stability
        self.memory = deque(maxlen=10000) # Increased memory
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # Slower decay for better exploration
        self.action_dim = action_dim
        self.tau = 0.005 # Soft update parameter

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def train_step(self, batch_size=64): # Larger batch size
        if len(self.memory) < batch_size: return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        actions = torch.LongTensor(actions)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            # Double DQN style: use model to pick action, target to get value
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Huber loss is more robust to reward outliers (like +50.0)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents catastrophic updates from big sparse rewards
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Soft update instead of hard sync
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def sync_target_model(self):
        """Used for manual sync if desired, but now redundant with soft updates."""
        self.target_model.load_state_dict(self.model.state_dict())