import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.constants import Constants as C

def decode_action(action: int):
    n_cards = action // (C.NUMBER_OF_CARDS_PER_SUIT+1)
    card_number = action % (C.NUMBER_OF_CARDS_PER_SUIT+1)
    print(f"To move: {C.N_CARDS_TO_TEXT[n_cards]} {str(card_number)}")