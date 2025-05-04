import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Callable, Optional
from collections import defaultdict

from agent.agent import Agent

from agent_evaluation.elo import DynamicEloSystem

class AgentTestingSystem(DynamicEloSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def select_test_opponents(self, model_pool: List[str], num_opponents: int = 5) -> List[str]:
        """
        Select diverse opponents for testing
        Strategy: Pick opponents across different rating ranges
        """
        if len(model_pool) <= num_opponents:
            return model_pool
            
        # Sort models by rating
        rated_agents = [(self.ratings[agent_id]['rating'], agent_id) for agent_id in model_pool]
        rated_agents.sort()
        
        if len(rated_agents) < num_opponents:
            return [m[1] for m in rated_agents]
            
        # Select diverse range of opponents
        step = (len(rated_agents) - 1) / (num_opponents - 1)
        indices = [int(i * step) for i in range(num_opponents)]
        return [rated_agents[i][1] for i in indices]
    
    def test_model(self, test_agent: Agent,num_opponents: int = 5) -> pd.DataFrame:
        """
        Test a single model against selected opponents
        
        Args:
            test_model: ID of the model to test
            model_pool: Dictionary of available models and their functions
            num_opponents: Number of opponents to test against
        """
        self.register_agent(test_agent)
        test_model_id = test_agent.model.id
        # Select diverse opponents
        available_opponents = [m for m in self.model_info.keys() if m != test_model_id]
        test_opponents = self.select_test_opponents(available_opponents, num_opponents)
        print("Test opponents are:", test_opponents)
        
        test_results = []
        
        # Play matches against each opponent
        for opponent in test_opponents:
            # Single match against each opponent
            print("Opponent: ", opponent)
            score = self.compare_performance(test_model_id, opponent)
            
            # Update ratings
            self.update_ratings(test_model_id, opponent, score)
            
            # Record match details
            match_detail = {
                'timestamp': datetime.now(),
                'test_model': test_model_id,
                'opponent': opponent,
                'score': score,
                'test_model_rating': self.ratings[test_model_id]['rating'],
                'opponent_rating': self.ratings[opponent]['rating']
            }
            
            test_results.append(match_detail)
            self.match_history.append(match_detail)
        
        return pd.DataFrame(test_results)
