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
writer = SummaryWriter()

class A2CScum:
    def __init__(self, number_of_agents, agent_pool: AgentPool=None, callback=None, load_checkpoints=False, **kwargs):
        self.env = ScumEnv(number_players=number_of_agents)
        self.agent_pool = AgentPool(number_of_agents, load_checkpoints, **kwargs) if agent_pool is None else agent_pool
        self.total_steps = 0
        self.ep_rewards = [[] for _ in range(number_of_agents)]
        self.last_mean_average_reward = [-np.inf for _ in range(5)]
        self.aggregate_states_every = C.AGGREGATE_STATS_EVERY
        self.callback = callback
    
    def learn(self, total_episodes, eval_every: int = 5000):
        from agent_evaluation.single_elo import AgentTestingSystem  # Move import here
 
        agent_testing = AgentTestingSystem(load_path="objects/agent_testing_system.pkl")
        
        for episode in tqdm(range(1, total_episodes + 1), ascii=True, unit='episodes'):
            episode_rewards = self.run_episode()
        
            for i, reward in enumerate(episode_rewards):
                self.ep_rewards[i].append(reward)

            if episode % eval_every == 0:
                ratings = []
                for i in range(C.NUMBER_OF_AGENTS):
                    agent = self.agent_pool.get_agent(i)
                    results = agent_testing.test_model(agent, num_opponents=C.NUMBER_OF_AGENTS)
                    print("Results of ELO are: ", results)

                    rating = agent_testing.get_model_summary(model_id=agent.model.id)['rating']
                    ratings.append(rating)
                max_rating = np.array(ratings).max()
                self.callback.last_mean_reward = max_rating
                print("Max ELO score is: ", str(max_rating))
                
                agent_testing.save_system(path="objects/agent_testing_system.pkl")

                
            if episode % self.aggregate_states_every == 0:
                average_rewards = []
                for i, average_reward in self.log_stats(episode):
                    average_rewards.append(average_reward)
                
                self.last_mean_average_reward = average_reward
                
                self.agent_pool.update_order(average_rewards)
                self.agent_pool.refresh_agents()
                self.agent_pool.save_agents()

            # Call the callback after each episode
            if self.callback is not None:
                if not self.callback._on_step():
                    break  # Stop training if the callback returns False

    def eval(self, episodes):
        ep_rewards = []
        for _ in range(episodes):
            ep_rewards.append(self.run_episode(eval=True))
        
        return ep_rewards

    def run_episode(self, eval: bool=False):
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
        
        if eval == False:
            for i in range(self.agent_pool.number_of_agents):
                agent = self.agent_pool.get_agent(i)
                results = agent.train(self.total_steps)
                print(results)
        

        
        return episode_rewards

    def log_stats(self, episode):
        for i in range(self.agent_pool.number_of_agents):
            recent_rewards = self.ep_rewards[i][-C.AGGREGATE_STATS_EVERY:]
            average_reward = sum(recent_rewards) / len(recent_rewards)
            min_reward = min(recent_rewards)
            max_reward = max(recent_rewards)
            print(f"Agent {i+1}: Avg: {average_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f} and current learning rate {self.agent_pool.get_agent(i).current_learning_rate:.6f}")
            writer.add_scalar(f"Reward/Agent {i+1}/Avg Reward", average_reward, episode)
            writer.flush()
            yield i, average_reward

    # def save_models(self, i: int) -> None:
    #     self.agent_pool.get_agent(i).save_model(path=f"models/checkpoints/agent_{i+1}.pt")
        

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

    
