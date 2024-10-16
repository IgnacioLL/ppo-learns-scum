import argparse
from dqn_agent import AgentPool
from constants import Constants as C
from env import ScumEnv
from tqdm import tqdm 
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def run_episode(env, agent_pool, total_steps) -> tuple[list[int], int]:
    finish_agents = [False] * 5
    episode_rewards = [0] * 5
    env.reset()
    print("Starting episode")
    print("_" * 100)

    while np.array(finish_agents).sum() != agent_pool.number_of_agents:
        print("_"*4)
        agent = agent_pool.get_agent(env.player_turn)
        action_state = env.get_cards_to_play()
        action = env.decide_move(action_state, epsilon=agent.epsilon, agent=agent)
        env._print_move(action)
        
        current_state, new_state, reward, finish, agent_number = env.step(action)
        finish_agents[agent_number] = finish
        episode_rewards[agent_number] += reward

        agent.update_replay_memory((current_state, action, reward, new_state, False))
        
        if finish:
            current_state, new_state, reward = env.get_stats_after_finish(agent_number=agent_number)
            episode_rewards[agent_number] += reward
            agent.update_replay_memory((current_state, action, reward, new_state, finish))
            finish_agents[agent_number] = finish

        td_error, tree_idxs = agent.train(agent_number, total_steps, finish)
        if td_error is not None:
            agent.buffer.update_priorities(tree_idxs, td_error)  
        total_steps += 1
        print("_"*4)

    return episode_rewards, total_steps

def log_stats(agent_pool: AgentPool, ep_rewards: list[list[int]], episode: int):
        for i in range(agent_pool.number_of_agents):
            recent_rewards = ep_rewards[i][-C.AGGREGATE_STATS_EVERY:]
            average_reward = sum(recent_rewards) / len(recent_rewards)
            min_reward = min(recent_rewards)
            max_reward = max(recent_rewards)
            print(f"Agent {i+1}: Avg: {average_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f} with epsilon {agent_pool.get_agent(i).epsilon} and learning rate {agent_pool.get_agent(i).current_learning_rate:.6f}")
            writer.add_scalar(f"Reward/Agent {i+1}/Avg Reward", average_reward, episode)
            writer.flush()
            yield i, average_reward

def save_models(agent_pool: AgentPool, i: int) -> None:
    agent_pool.get_agent(i).save_model(path=f"models/checkpoints/agent_{i+1}.pt")

def main(args):
    env = ScumEnv(args.number_of_agents)
    agent_pool = AgentPool(args.number_of_agents, load_checkpoints=args.load_checkpoints, learning_rate=args.learning_rate, epsilon=args.epsilon)
    ep_rewards = [[] for _ in range(args.number_of_agents)]
    total_steps = 0

    for episode in tqdm(range(1, args.episodes + 1), ascii=True, unit='episodes'):
        episode_rewards, total_steps = run_episode(env, agent_pool, total_steps)

        for i, reward in enumerate(episode_rewards):
            ep_rewards[i].append(reward)

        if episode % args.aggregate_stats_every == 0:
            average_rewards = []
            for i, average_reward in log_stats(agent_pool, ep_rewards, episode):
                average_rewards.append(average_reward)
                save_models(agent_pool, i)
            
            agent_pool.update_order(average_rewards)
            agent_pool.refresh_agents()
            agent_pool.save_agents()

        for agent in range(agent_pool.number_of_agents):
            agent_pool.get_agent(agent).decay_epsilon()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Scum Environment simulation")
    parser.add_argument("-n", "--number_of_agents", type=int, default=C.NUMBER_OF_AGENTS, help="Number of agents in the simulation")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate for the agents")
    parser.add_argument("-e", "--epsilon", type=float, default=0.1, help="Initial epsilon value for epsilon-greedy strategy")
    parser.add_argument("-lc", "--load_checkpoints", action="store_true", help="Load checkpoints if available")
    parser.add_argument("-ep", "--episodes", type=int, default=C.EPISODES, help="Number of episodes to run")
    parser.add_argument("-as", "--aggregate_stats_every", type=int, default=C.AGGREGATE_STATS_EVERY, help="Aggregate stats every N episodes")

    args = parser.parse_args()
    main(args)