import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np

from agent.agent_pool import AgentPool
from env.gymnasium_env import ScumEnv
from config.constants import Constants as C

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import argparse

from typing import List
from utils import logging


class A2CScum:
    def __init__(self, number_of_agents, agent_pool: AgentPool=None, callback=None, **kwargs):
        self.env = ScumEnv(number_players=number_of_agents)
        self.agent_pool = AgentPool(number_of_agents, **kwargs) if agent_pool is None else agent_pool
        self.total_steps = 0
        self.ep_rewards = []
        self.wins = []
        self.aggregate_stats_every = C.AGGREGATE_STATS_EVERY
        self.train_models_every = C.TRAIN_MODELS_EVERY
        self.create_checkpoint_every = C.CREATE_CHECKPOINT_EVERY
        self.assess_model = C.ASSESS_MODEL
        self.discount = C.DISCOUNT
        self.callback = callback
        self.writer = SummaryWriter()
    
    def learn(self, total_episodes):
        for episode in tqdm(range(1, total_episodes + 1), ascii=True, unit='episodes'):
            episode_rewards, win = self.env.run_episode(self.agent_pool, self.discount)

            self.append_rewards_to_historic_record(episode_rewards)
            self.append_win_to_historic_record(win)

            if episode == 100:
                self.agent_pool.flush_agents_buffers(episode)

            if episode % self.train_models_every == 0:
                training_performance_stats = self.train_models()
                logging.flush_performance_stats_tensorboard(self.writer, training_performance_stats, episode)

            if episode % self.assess_model == 0:
                win_rate = self.get_win_rate_last_n_episodes()
                logging.flush_average_win_rate_to_tensorboard(self.writer, win_rate, episode)
                if win_rate > .5:
                    self.agent_pool.refresh_agents_with_previous_executions()

            if episode % self.aggregate_stats_every == 0:
                win_rate = self.get_win_rate_last_n_episodes()
                logging.flush_average_win_rate_to_tensorboard(self.writer, win_rate, episode)
                average_reward = self.get_average_reward_last_n_episodes(C.AGGREGATE_STATS_EVERY)
                logging.flush_average_reward_to_tensorboard(self.writer, average_reward, episode)
            
            if episode % self.create_checkpoint_every == 0:
                self.agent_pool.save_models(episode)

    def train_models(self):
        performance_stats = {}
        for agent_number in range(self.agent_pool.number_of_agents):
            agent = self.agent_pool.get_agent(agent_number)
            if agent.training:
                performance_stats = agent.train(self.total_steps)
            else:
                agent.buffer.clear_buffer()
        return performance_stats

    def append_rewards_to_historic_record(self, episode_rewards: List[float]):
        for agent_number, reward in enumerate(episode_rewards):
            if self.agent_pool.get_agent(agent_number).training:
                self.ep_rewards.append(reward)

    def append_win_to_historic_record(self, win: bool):
        self.wins.append(win)

    def get_average_reward_last_n_episodes(self, n_episodes=C.AGGREGATE_STATS_EVERY) -> float:
        recent_rewards = self.ep_rewards[-n_episodes:]
        return sum(recent_rewards) / len(recent_rewards)

    def get_win_rate_last_n_episodes(self, n_episodes=C.AGGREGATE_STATS_EVERY) -> float:
        wins = self.wins[-n_episodes:].copy()
        return sum(wins) / len(wins)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Scum Environment simulation")
    parser.add_argument("-n", "--number_of_agents", type=int, default=C.NUMBER_OF_AGENTS, help="Number of agents in the simulation")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate for the agents")
    parser.add_argument("-lc", "--load_checkpoints", action="store_true", help="Load checkpoints if available")
    parser.add_argument("-ep", "--episodes", type=int, default=C.EPISODES, help="Number of episodes to run")

    args = parser.parse_args()

    model = A2CScum(number_of_agents=5, load_checkpoints=False, episodes=C.EPISODES)
    model.learn()

    
