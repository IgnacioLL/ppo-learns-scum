import collections
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from math import ceil

from constants import Constants as C
from utils import convert_to_binary_tensor, move_to_last_position

class ScumEnv:
    def __init__(self, number_players: int):
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
        self.previous_state = [
            torch.zeros(C.NUMBER_OF_POSSIBLE_STATES + self.number_players + C.NUMBER_OF_CARDS_PER_SUIT+1, dtype=torch.float32) 
            for _ in range(self.number_players)
            ]
        self.previous_reward = [0] * self.number_players
        self.previous_finish = [False] * self.number_players

        self.cards_thrown = [0] * (C.NUMBER_OF_CARDS_PER_SUIT + 1)

    def _compute_cards_x_player(self, cards: List[int]) -> int:
        number_of_cards = len(cards)
        number_of_cards_x_person = ceil((C.NUMBER_OF_SUITS*C.NUMBER_OF_CARDS_PER_SUIT)/self.number_players)
        return number_of_cards/number_of_cards_x_person
    
    def get_number_cards_x_person(self) -> List[int]:
        return [self._compute_cards_x_player(cards_player[0]) for cards_player in self.cards]

    def _get_deck(self) -> List[int]:
        deck = [rank for _ in range(C.NUMBER_OF_SUITS) for rank in range(1, C.NUMBER_OF_CARDS_PER_SUIT + 1)]
        deck.remove(C.NUMBER_OF_CARDS_PER_SUIT)
        deck.append(C.NUMBER_OF_CARDS_PER_SUIT + 1)  # 2 of hearts
        return deck

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

    @staticmethod
    def _extract_higher_order(cards: List[int], number_of_repetitions: int) -> List[int]:
        return list(set(item for item, count in collections.Counter(cards).items() if count >= number_of_repetitions))

    @classmethod
    def _get_combinations(cls, cards: List[int]) -> List[List[int]]:
        return [
            cards,
            cls._extract_higher_order(cards, 2),
            cls._extract_higher_order(cards, 3),
            cls._extract_higher_order(cards, 4)
        ]

    def reset(self) -> None:
        self.__init__(self.number_players)

    def _next_turn(self) -> int:
        next_player = (self.player_turn + 1) % self.number_players
        while not self.players_in_round[next_player]:
            next_player = (next_player + 1) % self.number_players
            if next_player == self.player_turn:
                break
        return next_player

    def _update_player_turn(self, skip: bool = False) -> None:
        turns_to_skip = 2 if skip else 1
        for _ in range(turns_to_skip):
            self.player_turn = self._next_turn()

    def _update_player_cards(self, card_number: int, n_cards: int) -> None:
        if card_number == C.NUMBER_OF_CARDS_PER_SUIT + 1:  # two of hearts
            self.cards[self.player_turn][0].remove(card_number)
        else:
            for _ in range(n_cards + 1):
                self.cards[self.player_turn][0].remove(card_number)
        self.cards[self.player_turn] = self._get_combinations(self.cards[self.player_turn][0])
        self.cards_thrown[card_number - 1] = (n_cards + 1) * (1/C.NUMBER_OF_SUITS)

    def _reinitialize_round(self) -> None:
        self.last_player = -1
        self.last_move = None
        self.players_in_round = self.players_in_game.copy()
        self.n_players_in_round = sum(self.players_in_round)

    def _check_players_playing(self) -> None:
        if self.n_players_in_round <= 1:
            if self.last_player != -1:  
                self.player_turn = self.last_player
            self._reinitialize_round()

    def _check_player_finish(self) -> Tuple[bool, int]:
        if not self.cards[self.player_turn][0]:
            self.players_in_game[self.player_turn] = False
            self.player_order[self.player_position_ending] = self.player_turn
            if self.player_position_ending == 0:
                finishing_reward = C.REWARD_WIN
            elif self.player_position_ending == 1:
                finishing_reward = C.REWARD_SECOND
            elif self.player_position_ending == self.number_players - 2:
                finishing_reward = C.REWARD_FOURTH
            elif self.player_position_ending == self.number_players - 1:
                finishing_reward = C.REWARD_LOSE
            else:
                finishing_reward = C.REWARD_THIRD
            
            self.player_position_ending += 1
            self.last_move = None
            self._reinitialize_round()
            return True, finishing_reward
        return False, 0

    def convert_to_binary_player_turn_cards(self) -> torch.tensor:
        return convert_to_binary_tensor(self.cards[self.player_turn], pass_option=True)

    def _get_cards_to_play(self) -> np.array:
        if self.last_move is None:
            return self._get_cards_to_play_init()
        else:
            return self._get_cards_to_play_followup()

    def _get_cards_to_play_init(self) -> torch.tensor:
        available_combinations = [i for i in range(C.NUMBER_OF_SUITS) if self.cards[self.player_turn][i]]
        return convert_to_binary_tensor(self.cards[self.player_turn], pass_option=False) ## the pass action which is not available in the first move

    def _get_cards_to_play_followup(self) -> torch.tensor:
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
        return convert_to_binary_tensor(cards, pass_option=True) ## add the pass action
    
    def get_cards_thrown(self):
        return self.cards_thrown
        
    def get_cards_to_play(self) -> torch.tensor:
        if sum(self.players_in_game) == 0:
            return

        if self.last_player == self.player_turn:
            self._reinitialize_round()
            return self.get_cards_to_play()

        action_space: torch.tensor = self._get_cards_to_play()

        players_info = self.get_number_cards_x_person()
        players_info = move_to_last_position(players_info, self.player_turn)
        cards_thrown = self.get_cards_thrown()

        state = torch.cat([action_space, torch.tensor(players_info), torch.tensor(cards_thrown)])
        
        return state

    def decide_move(self, state: torch.tensor, epsilon: float=1, agent: torch.nn.Module=None) -> int:
        action_space = state[:C.NUMBER_OF_POSSIBLE_STATES]
        if state is None:
            state = torch.tensor([0 for _ in range((C.NUMBER_OF_POSSIBLE_STATES - 1) + self.number_players + C.NUMBER_OF_CARDS_PER_SUIT+1)] + [1], dtype=torch.float32).to(C.DEVICE)
        if random.random() < epsilon:
            action_space_list = action_space.cpu().detach().numpy()
            indices = [i for i, x in enumerate(action_space_list) if x == 1]
            return random.choice(indices) + 1
        else:
            prediction = agent.predict(state, target=True)
            print("Using model to decide move")
            print("Prediction made by the model is: ", prediction[0].cpu().detach().numpy().round(2))
            
            # We set a large negative value to the masked predictions that are not possible
            masked_predictions = prediction[0] * action_space
            masked_predictions_npy  = masked_predictions.cpu().detach().numpy()

            masked_predictions_npy[masked_predictions_npy == 0] = float("-inf")
            
            return np.argsort(masked_predictions_npy)[-1] + 1  ## esto devolvera un valor entre 1 y 57 que sera la eleccion del modelo
    
    def _handle_unable_to_play(self) -> None:
        self.players_in_round[self.player_turn] = False
        self.n_players_in_round -= 1
        self._update_player_turn()
        self._check_players_playing()

    def step(self, action: int, current_state: torch.tensor) -> Tuple[torch.tensor, int, bool, int]:
        ## This will be the variables returned to compute the reward.
        ## We will use the past events to be able to compute the new states.
        agent_number = self.player_turn

        previous_state = self.previous_state[agent_number]
        previous_reward = self.previous_reward[agent_number]
        new_state = current_state ## this is the new state

        # Update previous state to the new state
        self.previous_state[agent_number] = new_state

        n_cards, card_number = self._decode_action(action)

        if self._is_pass_action(n_cards):
            return self._handle_pass_action(previous_state, new_state, previous_reward, False, agent_number)

        skip = self._is_skip_move(card_number)

        ## Delete the cards played from the player's hand also last move and last player to play.
        self._update_game_state(card_number, n_cards)
        
        ## Check if the player has finished the game and compute the reward
        finish, finishing_reward = self._check_player_finish()
        cards_reward = self._calculate_cards_reward(n_cards, finish)
        total_reward = cards_reward + finishing_reward

        ## Update the self.previous_reward and self.previous_finish
        self._update_previous_state(total_reward, finish)

        ## This changes self.player_turn
        self._update_player_turn(skip)

        return previous_state, new_state, previous_reward, finish, agent_number
    
    def get_stats_after_finish(self, agent_number: int) -> Tuple[int, int, int]:
        current_state = self.previous_state[agent_number]
        new_state = torch.zeros((C.NUMBER_OF_POSSIBLE_STATES) + self.number_players + C.NUMBER_OF_CARDS_PER_SUIT+1)
        reward = self.previous_reward[agent_number]
        return current_state, new_state, reward

    def _decode_action(self, action: int) -> Tuple[int, int]:
        n_cards = action // (C.NUMBER_OF_CARDS_PER_SUIT + 1)
        card_number = action % (C.NUMBER_OF_CARDS_PER_SUIT + 1)
        if card_number == 0:
            card_number = C.NUMBER_OF_CARDS_PER_SUIT + 1
        return n_cards, card_number

    def _is_pass_action(self, n_cards: int) -> bool:
        return n_cards == 4

    def _handle_pass_action(self, previous_state, new_state, previous_reward, previous_finish, agent_number):
        self.previous_reward[agent_number] = C.REWARD_PASS
        self._handle_unable_to_play()
        return previous_state, new_state, previous_reward, previous_finish, agent_number

    def _is_skip_move(self, card_number: int) -> bool:
        return self.last_move is not None and self.last_move[0] == card_number

    def _update_game_state(self, card_number: int, n_cards: int):
        self._update_player_cards(card_number, n_cards)
        self.last_move = [card_number, n_cards]
        self.last_player = self.player_turn

    def _calculate_cards_reward(self, n_cards: int, finish: bool) -> int:
        cards_reward = C.REWARD_CARD * (n_cards + 1)
        if finish:
            cards_reward += C.REWARD_EMPTY_HAND
        return cards_reward

    def _update_previous_state(self, total_reward: int, finish: bool):
        self.previous_reward[self.player_turn] = total_reward
        self.previous_finish[self.player_turn] = finish

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
        n_cards = action // (C.NUMBER_OF_CARDS_PER_SUIT+1)
        card_number = action % (C.NUMBER_OF_CARDS_PER_SUIT+1)
        
        print("Player to move is: ", self.player_turn)
        print("Last player to play was: ", self.last_player)
        print(f"Move made: {C.N_CARDS_TO_TEXT[n_cards]} {str(card_number)}")
    
    @staticmethod
    def _print_model_prediction(prediction: torch.tensor, masked_probabilities: torch.tensor) -> None:
        print("Prediction made by the model is: ", prediction)
        print("Masked probabilities are: ", masked_probabilities)
        print("-"*100)