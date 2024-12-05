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

writer = SummaryWriter()

class A2CScum:
    def __init__(self, number_of_agents, agent_pool: AgentPool=None, callback=None, load_checkpoints=False, **kwargs):
        self.env = ScumEnv(number_players=number_of_agents)
        self.agent_pool = AgentPool(number_of_agents, load_checkpoints, **kwargs) if agent_pool is None else agent_pool
        self.total_steps = 0
        self.ep_rewards = [[] for _ in range(number_of_agents)]
        self.aggregate_stats_every = C.AGGREGATE_STATS_EVERY
        self.callback = callback
    
    def learn(self, total_episodes):
        for episode in tqdm(range(1, total_episodes + 1), ascii=True, unit='episodes'):
            episode_rewards = self.run_episode(episode)

            self.append_rewards_to_historic_record(episode_rewards)

            if episode % self.aggregate_stats_every == 0:
                average_rewards = self.get_average_reward_last_n_episodes(episode)
                # self.agent_pool.swap_worst_models_for_best_ones(average_rewards)


    def eval(self, episodes):
        ep_rewards = []
        for _ in range(episodes):
            ep_rewards.append(self.run_episode(eval=True))
        
        return ep_rewards

    def run_episode(self, episode, eval: bool=False):
        done_agents = [False] * 5
        episode_rewards = [0] * 5
        self.env.reset()

        while np.array(done_agents).sum() != self.agent_pool.number_of_agents:
            agent = self.agent_pool.get_agent(self.env.player_turn)

            state = self.env.get_cards_to_play()
            action = agent.decide_move(state)
            current_state, new_state, reward, done, agent_number = self.env.step(action, state)

            done_agents[agent_number] = done
            episode_rewards[agent_number] += reward

            agent.save_in_buffer(current_state, reward, new_state)
            
            if done:
                current_state, new_state, reward = self.env.get_stats_after_done(agent_number=agent_number)
                episode_rewards[agent_number] += reward
                agent.save_in_buffer(current_state, reward, new_state)
                done_agents[agent_number] = done

            self.total_steps += 1
        
        if eval == False:
            training_performance_stats = self.train_models()
            self.flush_performance_stats_tensorboard(training_performance_stats, episode)

        return episode_rewards

    def train_models(self):
        performance_stats = {}
        for agent_number in range(self.agent_pool.number_of_agents):
            agent = self.agent_pool.get_agent(agent_number)
            performance_stats[agent_number] = agent.train(self.total_steps)
        return performance_stats

    def append_rewards_to_historic_record(self, episode_rewards: List[float]):
        for agent_number, reward in enumerate(episode_rewards):
            self.ep_rewards[agent_number].append(reward)

    def _compute_average_reward_last_n_episodes(self, n_episodes):
        agent_rewards = []
        for i in range(self.agent_pool.number_of_agents):
            recent_rewards = self.ep_rewards[i][-n_episodes:]
            average_reward = sum(recent_rewards) / len(recent_rewards)
            agent_rewards.append(average_reward)
        return agent_rewards

    def get_average_reward_last_n_episodes(self, n_episodes=C.AGGREGATE_STATS_EVERY) -> List[float]:
        return self._compute_average_reward_last_n_episodes(n_episodes)

    def flush_performance_stats_tensorboard(self, performance_stats: dict, episode: int) -> None:
        for agent_number in performance_stats:
            for stat in performance_stats[agent_number]:  
                writer.add_scalar(f"{stat}/Agent {int(agent_number)+1}", 
                                  performance_stats[agent_number][stat], episode)
        writer.flush()

    def flush_average_reward_to_tensorboard(self, average_rewards: list, episode: int):
        average_rewards_format = {agent_number : {"avg_reward": avg_reward} for agent_number, avg_reward in enumerate(average_rewards)}
        self.flush_performance_stats_tensorboard(average_rewards_format, episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Scum Environment simulation")
    parser.add_argument("-n", "--number_of_agents", type=int, default=C.NUMBER_OF_AGENTS, help="Number of agents in the simulation")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate for the agents")
    parser.add_argument("-e", "--epsilon", type=float, default=0.1, help="Initial epsilon value for epsilon-greedy strategy")
    parser.add_argument("-lc", "--load_checkpoints", action="store_true", help="Load checkpoints if available")
    parser.add_argument("-ep", "--episodes", type=int, default=C.EPISODES, help="Number of episodes to run")
    parser.add_argument("-as", "--aggregate_stats_every", type=int, default=C.AGGREGATE_STATS_EVERY, help="Aggregate stats every N episodes")

    args = parser.parse_args()

    model = A2CScum(number_of_agents=5, load_checkpoints=False, episodes=C.EPISODES)
    model.learn()

    
