import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agent.agent_pool import AgentPool
from environment.gymnasium_env import ScumEnv
from config.constants import Constants as C

from tqdm import tqdm
import argparse

class ScumTraining:
    def __init__(
            self,
            number_of_agents,
            agent_pool: AgentPool,
            create_checkpoint_every=C.CREATE_CHECKPOINT_EVERY
            ):
        
        self.env = ScumEnv(number_players=number_of_agents)
        self.agent_pool = agent_pool
        self.total_steps = 0
        self.ep_rewards = []
        self.wins = []
        self.aggregate_stats_every = C.AGGREGATE_STATS_EVERY
        self.train_models_every = C.TRAIN_MODELS_EVERY
        self.create_checkpoint_every = create_checkpoint_every
        self.assess_model = C.ASSESS_MODEL
        self.discount = C.DISCOUNT
    
    def learn(self, begin_episode=1, end_episodes=C.EPISODES):
        for episode in tqdm(range(begin_episode, end_episodes + 1), ascii=True, unit=' episode'):
            self.agent_pool.randomize_order()
            episode_rewards, winner_player = self.env.run_episode(self.agent_pool, self.discount)

            self.agent_pool.append_rewards_to_historic_record_to_each_agent(episode_rewards)
            self.agent_pool.append_win_to_historic_record_to_each_agent(winner_player)

            if (episode % self.train_models_every == 0) & (episode != begin_episode):
                self.train_models(episode)

            if (episode % self.aggregate_stats_every == 0) & (episode != begin_episode):
                self.agent_pool.flush_average_reward_to_tensorboard_from_each_agent(self.aggregate_stats_every, episode)
                self.agent_pool.flush_win_rate_to_tensorboard_from_each_agent(self.aggregate_stats_every, episode)
            
            if episode % self.create_checkpoint_every == 0:
                self.agent_pool.save_models(episode)

    def train_models(self, episode):
        performance_stats = {}
        for agent_number in range(self.agent_pool.number_of_agents):
            agent = self.agent_pool.get_agent(agent_number)
            if agent.training:
                performance_stats = agent.train(episode)
            else:
                agent.buffer.clear_buffer()
        return performance_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Scum Environment simulation")
    parser.add_argument("-n", "--number_of_agents", type=int, default=C.NUMBER_OF_AGENTS, help="Number of agents in the simulation")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate for the agents")
    parser.add_argument("-lc", "--load_checkpoints", action="store_true", help="Load checkpoints if available")
    parser.add_argument("-ep", "--episodes", type=int, default=C.EPISODES, help="Number of episodes to run")

    args = parser.parse_args()

    model = ScumTraining(number_of_agents=5, load_checkpoints=False, episodes=C.EPISODES)
    model.learn()

    
