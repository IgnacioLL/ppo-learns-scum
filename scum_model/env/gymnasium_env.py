import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.constants import Constants as C
import torch
from typing import List, Tuple
from math import ceil
import random
import collections

from utils.utils import move_to_last_position, convert_to_binary_tensor
from utils import data_utils, env_utils

from agent.agent_pool import AgentPool

class ScumEnv(gym.Env):
    def __init__(self, number_players):
        super(ScumEnv, self).__init__()

        self.number_players = number_players
        self.cards = self._deal_cards()
        self.player_turn = 0    
        self.last_player = -1
        self.last_move = None
        self.players_in_game = [True] * self.number_players
        self.players_in_round = self.players_in_game.copy()
        self.n_players_in_round = sum(self.players_in_round)
        self.player_position_ending = 0
        self.player_order = [-1] * self.number_players
        self.state = [
            torch.zeros(C.NUMBER_OF_POSSIBLE_STATES + self.number_players + C.NUMBER_OF_CARDS_PER_SUIT+1, dtype=torch.float32) 
            for _ in range(self.number_players)
            ]
        self.previous_reward = [0] * self.number_players
        self.previous_done = [False] * self.number_players

        self.cards_thrown = [0] * (C.NUMBER_OF_CARDS_PER_SUIT + 1)
        
        # Define action and observation space
        # For example, if you have discrete actions, use spaces.Discrete
        self.action_space = spaces.Box(low=0, high=1, shape=(C.NUMBER_OF_POSSIBLE_STATES + self.number_players + C.NUMBER_OF_CARDS_PER_SUIT+1, ))  # Example: 0 or 1 actions
        
        # Define observation space (e.g., continuous or discrete states)
        self.observation_space = spaces.Discrete(C.NUMBER_OF_POSSIBLE_STATES)
        
        # Set initial state
        self.done = False

        self.winner_player = None

    def run_episode(self, agent_pool: AgentPool, discount: float, verbose=False):
        done_agents = [False] * C.NUMBER_OF_AGENTS
        episode_rewards = [0] * C.NUMBER_OF_AGENTS
        all_rewards = [[] for _ in range(C.NUMBER_OF_AGENTS)]
        self.reset()
        while self.not_all_agents_done(agent_pool.number_of_agents, done_agents):
            agent = agent_pool.get_agent(self.player_turn)
            state = self.get_state()
            action_space = self.get_action_space()
            action, log_prob = agent.decide_move(state, action_space)
            if verbose:
                self._print_move(action)
            current_state, reward, done, agent_number = self.step(action, state)

            done_agents[agent_number] = done
            episode_rewards[agent_number] += reward
            all_rewards[agent_number].append(reward)

            agent.buffer.save_in_buffer(current_state, reward, action_space, action, log_prob)

        agent_pool.apply_discounted_returns_in_agents_buffer(all_rewards, discount)

        return episode_rewards, self.get_winner_player()
    
    @staticmethod
    def not_all_agents_done(number_of_agents: int, done_agents: List[bool]) -> bool:
        return np.array(done_agents).sum() != number_of_agents


    def reset(self) -> None:
        self.__init__(self.number_players)

    def render(self, mode='human'):
        """Render the environment (optional)"""
        print(f"Current State: {self.cards}")

    
    def step(self, action: int, current_state: torch.Tensor):
        agent_number = self.player_turn

        # Update previous state to the new state
        self.state[agent_number] = current_state

        n_cards, card_number = self._decode_action(action)

        if self._is_pass_action(n_cards):
            return self._handle_pass_action(current_state, agent_number)

        skip = self._is_skip_move(card_number)

        ## Delete the cards played from the player's hand also last move and last player to play.
        self._update_game_state(card_number, n_cards)
        
        ## Check if the player has done the game and compute the reward
        done, done_reward = self._check_player_done()
        cards_reward = self._calculate_cards_reward(action, done)
        total_reward = cards_reward + done_reward

        ## Update the self.previous_reward and self.previous_done
        self._update_previous_state(total_reward, done)

        ## This changes self.player_turn
        self._update_player_turn(skip)

        return current_state.detach(), total_reward, done, agent_number
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        n_cards = action // (C.NUMBER_OF_CARDS_PER_SUIT + 1)
        card_number = action % (C.NUMBER_OF_CARDS_PER_SUIT + 1)
        if card_number == 0:
            card_number = C.NUMBER_OF_CARDS_PER_SUIT + 1
        return n_cards, card_number

    def _is_pass_action(self, n_cards: int) -> bool:
        return n_cards == 4
    
    def _handle_pass_action(self, current_state, agent_number):
        self._handle_unable_to_play()
        return current_state, C.REWARD_PASS, False, agent_number

    def _handle_unable_to_play(self) -> None:
        self.players_in_round[self.player_turn] = False
        self.n_players_in_round -= 1
        self._update_player_turn()
        self._check_players_playing()
    
    def _update_player_turn(self, skip: bool = False) -> None:
        turns_to_skip = 2 if skip else 1
        for _ in range(turns_to_skip):
            self.player_turn = self._next_turn()

    def _next_turn(self) -> int:
        next_player = (self.player_turn + 1) % self.number_players
        while not self.players_in_round[next_player]:
            next_player = (next_player + 1) % self.number_players
            if next_player == self.player_turn:
                break
        return next_player


    def _check_players_playing(self) -> None:
        if self.n_players_in_round <= 1:
            if self.last_player != -1:  
                self.player_turn = self.last_player
            self._reinitialize_round()

    def _reinitialize_round(self) -> None:
        self.last_player = -1
        self.last_move = None
        self.players_in_round = self.players_in_game.copy()
        self.n_players_in_round = sum(self.players_in_round)

    def _is_skip_move(self, card_number: int) -> bool:
        return self.last_move is not None and self.last_move[0] == card_number
    

    def _update_game_state(self, card_number: int, n_cards: int):
        self._update_player_cards(card_number, n_cards)
        self.last_move = [card_number, n_cards]
        self.last_player = self.player_turn


    def _update_player_cards(self, card_number: int, n_cards: int) -> None:
        if card_number == C.NUMBER_OF_CARDS_PER_SUIT + 1:  # two of hearts
            self.cards[self.player_turn][0].remove(card_number)
        else:
            for _ in range(n_cards + 1):
                self.cards[self.player_turn][0].remove(card_number)
        self.cards[self.player_turn] = self._get_combinations(self.cards[self.player_turn][0])
        self.cards_thrown[card_number - 1] = (n_cards + 1) * (1/C.NUMBER_OF_SUITS)


    def _check_player_done(self) -> Tuple[bool, int]:
        if not self.cards[self.player_turn][0]:
            self.players_in_game[self.player_turn] = False
            self.player_order[self.player_position_ending] = self.player_turn
            if self.player_position_ending == 0:
                done_reward = C.REWARD_WIN
                self.winner_player = self.player_turn
            elif self.player_position_ending == 1:
                done_reward = C.REWARD_SECOND
            elif self.player_position_ending == self.number_players - 2:
                done_reward = C.REWARD_FOURTH
            elif self.player_position_ending == self.number_players - 1:
                done_reward = C.REWARD_LOSE
            else:
                done_reward = C.REWARD_THIRD
            
            self.player_position_ending += 1
            self.last_move = None
            self._reinitialize_round()
            return True, done_reward
        return False, 0
    
    def _calculate_cards_reward(self, action: int, done: bool) -> int:
        n_cards, card_number = self._decode_action(action)
        if card_number == 14:
            return C.REWARD_CARD + C.REWARD_EMPTY_HAND * int(done)
        else:
            return C.REWARD_CARD * (n_cards + 1) +  C.REWARD_EMPTY_HAND * int(done)
    
    def _update_previous_state(self, total_reward: int, done: bool):
        self.previous_reward[self.player_turn] = total_reward
        self.previous_done[self.player_turn] = done

    
    def get_state(self) -> torch.Tensor:
        if sum(self.players_in_game) == 0:
            return

        if self.last_player == self.player_turn:
            self._reinitialize_round()
            return self.get_state()

        cards = convert_to_binary_tensor(self.cards[self.player_turn], pass_option=True)

        players_info = self.get_number_cards_x_person()
        players_info = move_to_last_position(players_info, self.player_turn)
        cards_thrown = self.get_cards_thrown()

        compact_action_space = data_utils.compact_form_of_states(self.get_action_space())

        state = torch.cat([cards, compact_action_space.to(C.DEVICE), torch.tensor(players_info, device=C.DEVICE), torch.tensor(cards_thrown, device=C.DEVICE)])
        return state.detach()

    def get_action_space(self) -> torch.Tensor:
        if self.last_move is None:
            return self._get_cards_to_play_init()
        else:
            return self._get_cards_to_play_followup()

    def _get_cards_to_play_init(self) -> torch.Tensor:
        return convert_to_binary_tensor(self.cards[self.player_turn], pass_option=False).detach() ## the pass action which is not available in the first move

    def _get_cards_to_play_followup(self) -> torch.Tensor:
        n_cards = self.last_move[1]
        two_of_hearts = [C.NUMBER_OF_CARDS_PER_SUIT + 1] if C.NUMBER_OF_CARDS_PER_SUIT + 1 in self.cards[self.player_turn][0] else []
        possibilities = self.cards[self.player_turn][n_cards]

        if self.last_move[0] == 5:
            cards =  [cards for cards in possibilities if cards in [5, 6]]
        elif self.last_move[0] == 8:
            cards = [cards for cards in possibilities if cards <= self.last_move[0]]
        else:
            cards = [cards for cards in possibilities if cards >= self.last_move[0]] + two_of_hearts
        cards = [cards if index == n_cards else [] for index in range(4)]
        return convert_to_binary_tensor(cards, pass_option=True).detach() ## add the pass action

    def get_number_cards_x_person(self) -> List[int]:
        return [self._compute_cards_x_player(cards_player[0]) for cards_player in self.cards]

    def _compute_cards_x_player(self, cards: List[int]) -> int:
        number_of_cards = len(cards)
        number_of_cards_x_person = ceil((C.NUMBER_OF_SUITS*C.NUMBER_OF_CARDS_PER_SUIT)/self.number_players)
        return number_of_cards/number_of_cards_x_person
    

    def get_cards_thrown(self):
        return self.cards_thrown


    def _deal_cards(self) -> List[List[List[int]]]:
        deck = self._get_deck()
        sampled_data = random.sample(deck, C.NUMBER_OF_CARDS)
        
        cards_per_player = C.NUMBER_OF_CARDS // self.number_players
        remainder = C.NUMBER_OF_CARDS % self.number_players
        
        cards_of_players = []
        for i in range(self.number_players):
            start = i * cards_per_player + min(i, remainder)
            end = start + cards_per_player + (1 if i < remainder else 0)
            player_cards = sorted(sampled_data[start:end])
            cards_of_players.append(self._get_combinations(player_cards))

        return cards_of_players

    def _get_deck(self) -> List[int]:
        deck = [rank for _ in range(C.NUMBER_OF_SUITS) for rank in range(1, C.NUMBER_OF_CARDS_PER_SUIT + 1)]
        deck.remove(C.NUMBER_OF_CARDS_PER_SUIT)
        deck.append(C.NUMBER_OF_CARDS_PER_SUIT + 1)  # 2 of hearts
        return deck
    

    @classmethod
    def _get_combinations(cls, cards: List[int]) -> List[List[int]]:
        return [
            cards,
            cls._extract_higher_order(cards, 2),
            cls._extract_higher_order(cards, 3),
            cls._extract_higher_order(cards, 4)
        ]


    @staticmethod
    def _extract_higher_order(cards: List[int], number_of_repetitions: int) -> List[int]:
        return list(set(item for item, count in collections.Counter(cards).items() if count >= number_of_repetitions))

    def get_stats_after_done(self, agent_number: int) -> Tuple[int, int, int]:
        current_state = self.state[agent_number]
        reward = self.previous_reward[agent_number]
        return current_state, reward

    def _print_game_state(self) -> None:
        if self.last_move is not None:
            print(f"Last move was: {C.N_CARDS_TO_TEXT[self.last_move[1]]} {str(self.last_move[0])}, played by {self.last_player}")
        else:
            print("Beginning of round")
        print("The player who has to move is: ", self.player_turn)
    
    def _print_players_in_game(self) -> None:
        players_playing = ", ".join([str(i) for i, x in enumerate(self.players_in_game) if x])
        print(f"Players playing are: {players_playing}")

    def _print_move(self, action: int) -> None:
        print("Player to move is: ", self.player_turn)
        print("Last player to play was: ", self.last_player)
        env_utils.decode_action(action)
    
    @staticmethod
    def _print_model_prediction(prediction: torch.Tensor, masked_probabilities: torch.Tensor) -> None:
        print("Prediction made by the model is: ", prediction)
        print("Masked probabilities are: ", masked_probabilities)
        print("-"*100)

    
    def get_winner_player(self) -> List[int]:
        winner_player = np.zeros(self.number_players).tolist()
        winner_player[self.winner_player] = 1
        return winner_player
