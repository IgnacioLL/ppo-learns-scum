import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.constants import Constants as C

def decode_action(action: int):
    n_cards = action // (C.NUMBER_OF_CARDS_PER_SUIT+1)
    card_number = action % (C.NUMBER_OF_CARDS_PER_SUIT+1)
    print(f"To move: {C.N_CARDS_TO_TEXT[n_cards]} {str(card_number)}")

def decode_action_for_api(action: int) -> dict:
    """
    Decodes an action index into a structured dictionary for API responses.
    
    Args:
        action: The action index (1-based)
    
    Returns:
        A dictionary with action details
    """
    # Handle pass action
    if action == C.NUMBER_OF_SUITS * (C.NUMBER_OF_CARDS_PER_SUIT + 1) + 1:
        return {
            "type": "pass",
            "description": "Pass"
        }
    
    # Adjust to 0-based for calculation
    action_0_based = action - 1
    
    # Calculate number of cards and card value
    n_cards = action_0_based // (C.NUMBER_OF_CARDS_PER_SUIT + 1) + 1
    card_value = action_0_based % (C.NUMBER_OF_CARDS_PER_SUIT + 1) + 1
    
    # Map card value to face name
    face_map = {1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 
                8: '10', 9: 'J', 10: 'Q', 11: 'K', 12: 'A', 13: '2', 14: '2H'}
    
    card_face = face_map.get(card_value, str(card_value))
    
    return {
        "type": "play",
        "cardValue": int(card_value),
        "cardFace": card_face,
        "count": int(n_cards),
        "description": f"Play {n_cards}x {card_face}"
    }