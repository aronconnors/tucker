import collections
import cv2
import gym
import numpy as np
from PIL import Image
import torch
from game import SnakeGameAI

#class DQNBreakout(gym.Wrapper):
class DQNBreakout:
    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        self.env = SnakeGameAI()

        #super(DQNBreakout, self).__init__(env)
        
        self.image_shape = (84, 84)
        self.repeat = repeat
        #self.lives = env.ale.lives()
        self.frame_buffer = []
        self.device = device

    def step(self, action):
        total_reward = 0
        done = False

        for i in range(self.repeat):
            observation, reward, done, score = self.env.play_step(action)

            total_reward += reward

            #current_lives = info['lives']

            '''if current_lives < self.lives:
                total_reward = total_reward -1
                self.lives = current_lives'''
            

            self.frame_buffer.append(observation)

            if done:
                break

        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1)
        done = done.to(self.device)

        return max_frame, total_reward, done, score
    
    def reset(self):
        self.frame_buffer = []

        observation = self.env.reset()

        #self.lives = self.env.ale.lives()

        observation = self.process_observation(observation)

        return observation
    
    def process_observation(self, observation):
        
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255

        img = img.to(self.device)

        return img