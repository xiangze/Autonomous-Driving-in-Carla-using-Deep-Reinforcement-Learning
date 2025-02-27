import os
import numpy as np

import torch
import torch.nn as nn
from encoder_init import EncodeState
from networks.on_policy.grpo.grpo import PolicyNet
from parameters import  *
from collections import deque

device = torch.device("cpu")

class Buffer:
    def __init__(self):
         # Batch data
        self.observation = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []

    def clear(self):
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]

class GRPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        #self.env = env
        self.obs_dim = 100
        self.action_dim = 2
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 7
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        self.town = town

        self.traj_per_update=20
        self.beta= 0.1
        self.eps =0.1
        self.checkpoint_file_no = 0
        
        self.policy = PolicyNet(self.obs_dim, self.action_dim, self.action_std) #

        self.optimizer = torch.optim.Adam([{'params': self.policy.net.parameters(), 'lr': self.lr}])

        self.old_policy = PolicyNet(self.obs_dim, self.action_dim, self.action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def get_action(self, obs, train):

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, logprob = self.old_policy.get_action_and_log_prob(obs.to(device))
        if train:
            self.memory.observation.append(obs.to(device))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)

        return action.detach().cpu().numpy().flatten()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std

    def update(self,loss):
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

#from https://superb-makemake-3a4.notion.site/group-relative-policy-optimization-GRPO-18c41736f0fd806eb39dc35031758885
    def collect_trajectory(self,env,num_trajs=200):
        observation = env.reset()
        log_probs = []
        observations = []
        chosen_actions = []
        episode_reward = 0

        for t in range(num_trajs):
            observations.append(observation)
            logits = self.policy.net(torch.from_numpy(observation).float())
            probs = torch.nn.functional.softmax(logits, dim=0)
            action = torch.multinomial(probs, 1).item()

            observation, reward, done, _ = env.step(action)
            log_prob = torch.log(probs[action])
            log_probs.append(log_prob.item())
            chosen_actions.append(action)
            episode_reward += reward

            if done:
                break

        normalized_reward = episode_reward / num_trajs
        return observations, log_probs, chosen_actions, normalized_reward
        
    def clip(self,r):
        return  torch.clamp(r, min=1 - self.eps, max=1 + self.eps)                   
    
    def normalize(self,rewards):
        mean_reward = sum(rewards) / len(rewards)
        std_reward = np.std(rewards) + 1e-8
        return [(r - mean_reward) / std_reward for r in rewards]
    
    def learn1(self,env=None,num_ite=20):

        trajs=[ self.collect_trajectory(env) for i in range(self.traj_per_update)]
        rewards = [r for o, l, a, r in trajs]
        advantages = self.normalize(rewards)

        for _ in range(num_ite):
            loss = 0
            # each trajectory in the group
            for traj, advantage in zip(trajs, advantages):
                (observations, log_probs, chosen_actions, _) = traj
                trajectory_loss = 0
                # iterating over each time step in the trajectory
                for t in range(len(observations)):
                    new_policy_probs = torch.nn.functional.softmax(self.policy.net(torch.from_numpy(observations[t]).float()), dim=0)
                    new_log_probs = torch.log(new_policy_probs)[chosen_actions[t]]
                    ratio = torch.exp(new_log_probs - log_probs[t])
                    trajectory_loss += -self.clip(ratio)* advantage
                trajectory_loss /= len(observations)
                loss += trajectory_loss
            loss /= len(trajs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()        
    
        self.episode_rewards.extend(rewards)
        avg_reward = sum(self.episode_rewards)/len(self.episode_rewards)

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

        return avg_reward

    
    def save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def chkpt_save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        if self.checkpoint_file_no !=0:
            self.checkpoint_file_no -=1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)
   
    def load(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2]) - 1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))
            