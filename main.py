import torch
import numpy as np

from a2c_agent import A2C_Agent, Pool_A2C_Agents
from env import ScumEnv
from constants import Constants as C

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import argparse

import optuna


writer = SummaryWriter()



class A2C_Scum:
    def __init__(self, number_of_agents, load_checkpoints, episodes):
        self.env = ScumEnv(number_players=number_of_agents)
        self.agent_pool = Pool_A2C_Agents(number_of_agents, load_checkpoints)
        self.total_steps = 0
        self.ep_rewards = [[] for _ in range(args.number_of_agents)]
        self.episodes = episodes
        self.last_mean_average_reward = [-np.inf for _ in range(5)]
        self.aggregate_states_every = C.AGGREGATE_STATS_EVERY
    
    def learn(self):
        for episode in tqdm(range(1, self.episodes + 1), ascii=True, unit='episodes'):
            episode_rewards = self.step(self.env, self.agent_pool, self.total_steps)
        
            for i, reward in enumerate(episode_rewards):
                self.ep_rewards[i].append(reward)

            if episode % self.aggregate_stats_every == 0:
                average_rewards = []
                for i, average_reward in self.log_stats(self.agent_pool, self.ep_rewards, episode):
                    average_rewards.append(average_reward)
                    self.save_models(self.agent_pool, i)
                
                self.last_mean_average_reward = average_reward.copy()
                
                self.agent_pool.update_order(average_rewards)
                self.agent_pool.refresh_agents()
                self.agent_pool.save_agents()

    def step(self):
        finish_agents = [False] * 5
        episode_rewards = [0] * 5
        self.env.reset()
        print("Starting episode")
        print("_" * 100)

        while np.array(finish_agents).sum() != self.agent_pool.number_of_agents:
            agent = self.agent_pool.get_agent(self.env.player_turn)

            state = self.env.get_cards_to_play()
            action, rw_decision = agent.decide_move(state)
            current_state, new_state, reward, finish, agent_number = self.env.step(action, state)

            reward = reward + rw_decision
            finish_agents[agent_number] = finish
            episode_rewards[agent_number] += reward

            agent.memory.append((current_state, reward, new_state))
            
            if finish:
                current_state, new_state, reward = self.env.get_stats_after_finish(agent_number=agent_number)
                episode_rewards[agent_number] += reward
                agent.memory.append((current_state, reward, new_state))
                finish_agents[agent_number] = finish

            self.total_steps += 1
        
        agent.train(self.total_steps)

        return episode_rewards

    def log_stats(agent_pool: Pool_A2C_Agents, ep_rewards: list[list[int]], episode: int):
        for i in range(agent_pool.number_of_agents):
            recent_rewards = ep_rewards[i][-C.AGGREGATE_STATS_EVERY:]
            average_reward = sum(recent_rewards) / len(recent_rewards)
            min_reward = min(recent_rewards)
            max_reward = max(recent_rewards)
            print(f"Agent {i+1}: Avg: {average_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f} and current learning rate {agent_pool.get_agent(i).current_learning_rate:.6f}")
            writer.add_scalar(f"Reward/Agent {i+1}/Avg Reward", average_reward, episode)
            writer.flush()
            yield i, average_reward

    def save_models(agent_pool: Pool_A2C_Agents, i: int) -> None:
        agent_pool.get_agent(i).save_model(path=f"models/checkpoints/agent_{i+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Scum Environment simulation")
    parser.add_argument("-n", "--number_of_agents", type=int, default=C.NUMBER_OF_AGENTS, help="Number of agents in the simulation")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate for the agents")
    parser.add_argument("-e", "--epsilon", type=float, default=0.1, help="Initial epsilon value for epsilon-greedy strategy")
    parser.add_argument("-lc", "--load_checkpoints", action="store_true", help="Load checkpoints if available")
    parser.add_argument("-ep", "--episodes", type=int, default=C.EPISODES, help="Number of episodes to run")
    parser.add_argument("-as", "--aggregate_stats_every", type=int, default=C.AGGREGATE_STATS_EVERY, help="Aggregate stats every N episodes")

    args = parser.parse_args()

    
