import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Callable, Optional
import heapq
from collections import defaultdict


from agent.a2c_agent import A2CAgent
from agent.agent_pool import AgentPool
from agent.a2c_scum import A2CScum

from config.constants import Constants as C

import pickle as pkl

class DynamicEloSystem:
    def __init__(self, k_factor=32, initial_rating=1500, load_path: Optional[str] = None):
        if load_path:
            self._load_system(load_path)
        else:
            self.k_factor = k_factor
            self.initial_rating = initial_rating
            self.ratings = defaultdict(lambda: {
                'rating': initial_rating,
                'matches': 0,
                'last_match': None,
                'uncertainty': 1.0  # Higher means more uncertain
            })
            self.match_history = []
            self.active_matches = set()
            self.number_of_models = 0
            self.model_info = {}

    def save_system(self, path: str) -> None:
        """
        Save the current state of the ELO system to disk.
        
        Args:
            path (str): Path where the system state should be saved
        """
        # Convert defaultdict to regular dict for serialization
        ratings_dict = dict(self.ratings)
        
        system_state = {
            'k_factor': self.k_factor,
            'initial_rating': self.initial_rating,
            'ratings': ratings_dict,
            'match_history': self.match_history,
            'active_matches': list(self.active_matches),  # Convert set to list for serialization
            'number_of_models': self.number_of_models,
            'model_info': self.model_info
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pkl.dump(system_state, f)

    def _load_system(self, path: str) -> None:
        """
        Load the ELO system state from disk.
        
        Args:
            path (str): Path to the saved system state
        """
        with open(path, 'rb') as f:
            system_state = pkl.load(f)
        
        self.k_factor = system_state['k_factor']
        self.initial_rating = system_state['initial_rating']
        
        # Recreate defaultdict with the loaded ratings
        self.ratings = defaultdict(lambda: {
            'rating': self.initial_rating,
            'matches': 0,
            'last_match': None,
            'uncertainty': 1.0
        })
        self.ratings.update(system_state['ratings'])
        
        self.match_history = system_state['match_history']
        self.active_matches = set(system_state['active_matches'])  # Convert list back to set
        self.number_of_models = system_state['number_of_models']
        self.model_info = system_state['model_info']

        
    def register_agent(self, agent: A2CAgent) -> int:
        """Register a new model in the system"""
        model_id = agent.model.id
        if agent not in self.ratings:
            self.ratings[model_id] = {
                'rating': self.initial_rating,
                'matches': 0,
                'last_match': None,
                'uncertainty': 1.0
            }


        path = f"{C.MODELS_PATH}/{agent.model.size}_{agent.model.id}.pt"
        agent.save_model(path=path)

        self.model_info[model_id] = {'path': path, 'size': agent.model.size} 

    
    def select_matches(self, available_models: List[str], num_matches: int) -> List[tuple]:
        """
        Select optimal matches based on ratings and uncertainties
        Uses a modified Swiss system with uncertainty weighting


        """
        matches = []
        model_data = []
        
        # Create sorted list of models by rating
        for agent in available_models:
            
            heapq.heappush(model_data, (
                -self.ratings[agent]['uncertainty'],  # Prioritize uncertain models
                -self.ratings[agent]['rating'],
                agent
            ))
        
        while len(matches) < num_matches and len(model_data) >= 2:
            # Get models with highest uncertainty
            try:
                _, _, model_a = heapq.heappop(model_data)
                _, _, model_b = heapq.heappop(model_data)
                
                # Check if this pair is already playing
                # TODO: Add the __lt__ method in A2CAgent using the id which will also need to be added.
                pair = tuple(sorted([model_a, model_b]))
                if pair not in self.active_matches:
                    matches.append(pair)
                    self.active_matches.add(pair)
            except IndexError:
                break
                
        return matches
    
    def update_ratings(self, model_a: str, model_b: str, score: float) -> None:
        """Update ratings after a match"""
        # Remove from active matches
        # self.active_matches.remove(tuple(sorted([model_a, model_b])))
        
        # Get current ratings
        rating_a = self.ratings[model_a]['rating']
        rating_b = self.ratings[model_b]['rating']
        
        # Calculate expected scores
        expected_a = self._expected_score(rating_a, rating_b)
        
        # Calculate K-factor modifications based on uncertainty
        k_a = self.k_factor * self.ratings[model_a]['uncertainty']
        k_b = self.k_factor * self.ratings[model_b]['uncertainty']
        
        # Update ratings
        new_rating_a = rating_a + k_a * (score - expected_a)
        new_rating_b = rating_b + k_b * ((1 - score) - (1 - expected_a))
        
        # Update model data
        now = datetime.now()
        for model, new_rating, k in [(model_a, new_rating_a, k_a), 
                                   (model_b, new_rating_b, k_b)]:
            self.ratings[model].update({
                'rating': new_rating,
                'matches': self.ratings[model]['matches'] + 1,
                'last_match': now,
                'uncertainty': max(0.1, self.ratings[model]['uncertainty'] * 0.95)  # Decay uncertainty
            })
        
        # Record match
        self.match_history.append({
            'timestamp': now,
            'model_a': model_a,
            'model_b': model_b,
            'score': score,
            'rating_a': new_rating_a,
            'rating_b': new_rating_b
        })
    
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using logistic function"""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def run_dynamic_tournament(self, 
                             performance_functions: Dict[int, A2CAgent],
                             matches_per_round: Optional[int] = None,
                             total_rounds: int = 16) -> pd.DataFrame:
        """
        Run a tournament with a dynamic pool of models
        
        Args:
            performance_functions: Dict mapping model IDs to their performance functions
            matches_per_round: Number of concurrent matches per round (default: n/2 where n is pool size)
            total_rounds: Total number of tournament rounds
        """
        if matches_per_round is None:
            matches_per_round = len(performance_functions) // 2
        
        results = []
        
        for round_num in range(total_rounds):
            # Select matches for this round
            available_models = list(performance_functions.keys())
            matches = self.select_matches(available_models, matches_per_round)
            
            # Run matches
            for model_a, model_b in matches:
                score = self.compare_performance(
                    performance_functions[model_a],
                    performance_functions[model_b]
                )
                self.update_ratings(model_a, model_b, score)
                
            # Record round results
            round_ratings = self.get_current_rankings()
            round_ratings['round'] = round_num
            results.append(round_ratings)
            
        return pd.concat(results, ignore_index=True)
    
    def compare_performance(self, agent1_id: str, agent2_id: str, total_num_episodes: int = 200) -> float:
        """Compare models"""
        path_agent1 = self.model_info[agent1_id]['path']
        path_agent2 = self.model_info[agent2_id]['path']

        size_agent1 = self.model_info[agent1_id]['size']
        size_agent2 = self.model_info[agent2_id]['size']


        agent1 = A2CAgent(model=size_agent1, path=path_agent1, model_id=agent1_id)
        agent2 = A2CAgent(model=size_agent2, path=path_agent2, model_id=agent2_id)

        scores_agent1 = []
        scores_agent2 = []
        agents = [agent1, agent2]
        for i in range(C.NUMBER_OF_AGENTS):
            agent_pool = AgentPool(num_agents=C.NUMBER_OF_AGENTS)
            for a in range(C.NUMBER_OF_AGENTS):
                for j in range(2):
                    if i == a:
                        agent_pool.set_agent(agent=agents[j], agent_number=a)
                    else:
                        agent_pool.set_agent(agent=agents[-(j+1)], agent_number=a)
                    env = A2CScum(number_of_agents=C.NUMBER_OF_AGENTS, agent_pool=agent_pool)

                    for _ in range(total_num_episodes//(C.NUMBER_OF_AGENTS*2)):
                        episode_rewards = env.run_episode(eval=True)
                        agent_performance = episode_rewards[i] == np.array(episode_rewards).max()
                        win = float(agent_performance > 0)
                        scores_agent1.append(win) if j == 0 else scores_agent2.append(win)

        ## Who won more?
        print("Score agent 1:", scores_agent1)
        print("Score agent 2:", scores_agent2)
        score = np.mean(scores_agent1) / (np.mean(scores_agent2) + np.mean(scores_agent1))
                    
        return score
        
    def get_current_rankings(self) -> pd.DataFrame:
        """Get current rankings with additional statistics"""
        rankings = []
        for model, data in self.ratings.items():
            rankings.append({
                'model': model,
                'rating': data['rating'],
                'matches_played': data['matches'],
                'uncertainty': data['uncertainty'],
                'last_match': data['last_match']
            })
        
        df = pd.DataFrame(rankings)
        if not df.empty:
            return df.sort_values('rating', ascending=False).reset_index(drop=True)
        return df
    
    def get_model_history(self, model_id: str) -> pd.DataFrame:
        """Get match history for a specific model"""
        history = []
        for match in self.match_history:
            if model_id in (match['model_a'], match['model_b']):
                is_model_a = model_id == match['model_a']
                history.append({
                    'timestamp': match['timestamp'],
                    'opponent': match['model_b'] if is_model_a else match['model_a'],
                    'score': match['score'] if is_model_a else 1 - match['score'],
                    'rating': match['rating_a'] if is_model_a else match['rating_b']
                })
        return pd.DataFrame(history)
    
    def get_model_summary(self, model_id: str) -> Dict:
        """Get summary statistics for a model"""

        return {
            'model_id': model_id,
            'rating': self.ratings[model_id]['rating'],
            'matches_played':  self.ratings[model_id]['matches'],
            'uncertainty': self.ratings[model_id]['uncertainty']
        }
    
