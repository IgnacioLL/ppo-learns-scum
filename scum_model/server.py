# server.py
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(project_root)
# --- End Path Addition ---

try:
    from agent.agent_pool import AgentPool
    from env.gymnasium_env import ScumEnv
    from config.constants import Constants as C
    from utils import env_utils
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure server.py is in the correct location relative to 'agent', 'env', 'config', 'utils'")
    print(f"Current sys.path: {sys.path}")
    exit(1)

app = Flask(__name__)
CORS(app)

# --- Global Game State ---
game_env: ScumEnv = None
agent_pool: AgentPool = None

# --- Card Mapping (Keep as is) ---
FACE_MAP_TO_UI = {1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q', 11: 'K', 12: 'A', 13: '2', 14: '2'}
SUITS = ['Heart', 'Spade', 'Club', 'Diamond']

# --- Helper Functions (Keep map_card_to_ui, map_hand_to_ui, map_played_cards_to_ui as is) ---
def map_card_to_ui(card_value: int, card_index: int) -> dict:
    """Maps an internal card representation to the format React expects."""
    face = FACE_MAP_TO_UI.get(card_value, '?')
    suit = 'Spade'
    return {
        "id": f"{face}-{suit}-{random.random()}", # Generate a unique ID for React key
        "cardFace": face,
        "suit": suit,
        "value": card_value # The crucial part for logic/sorting
    }

def map_hand_to_ui(hand_values: list) -> list:
    """Maps a list of internal card values to UI card objects for a player's hand."""
    ui_hand = []
    # Sort by value for consistent display in hand
    for i, value in enumerate(sorted(hand_values)):
        face = FACE_MAP_TO_UI.get(value, '?')
        suit = 'Spade' # Example: Make Aces always Spade
        if value == 14: # Special suit for '2' maybe?
            suit = 'Heart' # Example: Make 2s always Heart

        ui_hand.append({
            "id": f"{face}-{suit}-{i}-{random.random()}", # Ensure unique ID
            "cardFace": face,
            "suit": suit,
            "value": value
        })
    return ui_hand

def map_pile_to_ui(pile_values: list) -> list:
    """Maps a list of card values (the current pile) to UI card objects."""
    if not pile_values:
        return []

    ui_cards = []
    # We don't necessarily know the original suits, so assign them
    # consistently based on value or index for display. Value is most important.
    for i, value in enumerate(pile_values):
        face = FACE_MAP_TO_UI.get(value, '?')
        # Assign suit based on index in pile for visual variety
        suit = 'Spade'
        if value == 14: suit = 'Heart' # Keep special suits if desired

        ui_cards.append({
            "id": f"table-{face}-{suit}-{i}-{random.random()}", # Unique ID for table cards
            "cardFace": face,
            "suit": suit,
            "value": value
        })
    return ui_cards

def map_played_cards_to_ui(last_move: list) -> list:
    """Maps the internal last_move [card_value, n_cards_index] to UI cards."""
    if not last_move:
        return []
    card_value, n_cards_index = last_move
    num_cards = n_cards_index + 1 # n_cards_index is 0 for 1 card, 1 for 2, etc.
    ui_cards = []
    face = FACE_MAP_TO_UI.get(card_value, '?')
    for i in range(num_cards):
        suit = 'Spade'
        ui_cards.append({
            "id": f"table-{face}-{suit}-{i}-{random.random()}",
            "cardFace": face,
            "suit": suit,
            "value": card_value
        })
    return ui_cards

def get_game_state_for_ui(env: ScumEnv, message: str = "", phase: str = None) -> dict:
    """Constructs the game state dictionary for the React frontend."""
    if not env:
        return {"error": "Game not initialized"}

    players_ui = []
    for i in range(env.number_players):
        raw_hand = env.cards[i][0] if env.cards and len(env.cards) > i and env.cards[i] else []
        ui_hand = map_hand_to_ui(raw_hand)
        is_still_in_round = False
        if i < len(env.players_in_game) and i < len(env.players_in_round):
             is_still_in_round = env.players_in_game[i] and env.players_in_round[i]
        # --------------------------------------------------------------------------

        players_ui.append({
            "id": f"player-{i}",
            "name": f"Player {i}" if i != 0 else "You",
            "hand": ui_hand,
            "cardCount": len(raw_hand),
            "isHuman": i == 0,
            "finishedRank": env.player_order.index(i) + 1 if i in env.player_order else None,
            "isStillInRound": is_still_in_round,
        })

    # Determine game phase if not explicitly provided
    current_phase = phase
    if current_phase is None:
        # Check if the game/round is over (only 1 or 0 players left *in the game*)
        if sum(env.players_in_game) <= 1:
            current_phase = 'roundOver' # Or potentially 'gameOver' if you implement multi-round logic
        else:
            current_phase = 'playing'

    # Determine game message
    current_message = message
    if not message:
        if current_phase == 'playing':
            # Ensure player_turn is valid before accessing players_ui
            if 0 <= env.player_turn < len(players_ui):
                 current_message = f"{players_ui[env.player_turn]['name']}'s turn."
                 # Add specific message if player is not in round (should pass automatically?)
                 # This might already be handled by the AI/human logic skipping turns
                 if not env.players_in_round[env.player_turn]:
                     current_message += " (Must Pass)" # Optional clarification
            else:
                 current_message = "Waiting for next turn..." # Fallback
        elif current_phase == 'roundOver':
            winner_idx = env.player_order[0] if env.player_order and len(env.player_order) > 0 and env.player_order[0] != -1 else -1
            winner_name = f"Player {winner_idx}" if winner_idx != -1 else "Someone"
            current_message = f"Round Over! {winner_name} finished first!" # Adjusted message
        # Add gameOver message if needed

    current_pile_ui = []
    if hasattr(env, 'current_pile'): # Check if the attribute exists
        current_pile_ui = map_pile_to_ui(env.current_pile)

    return {
        "gamePhase": current_phase,
        "players": players_ui,
        "cardsOnTable": current_pile_ui,
        "currentPlayerIndex": env.player_turn if current_phase == 'playing' else -1,
        "gameMessage": current_message,
        "humanPlayerId": "player-0",
        # No need to add activePlayerCount here, the frontend calculates it
    }

# --- Action Conversion Helpers (Keep find_action_index, get_pass_action_index as is) ---
def find_action_index(env: ScumEnv, cards_to_play: list) -> int:
    """
    Converts a list of UI card objects into the corresponding action index
    for env.step().
    """
    if not cards_to_play:
        return -1

    card_value = cards_to_play[0]['value']
    n_cards = len(cards_to_play)
    n_cards_index = n_cards - 1

    action_space_tensor = env.get_action_space()
    internal_card_number = card_value
    if internal_card_number == 0: internal_card_number = C.NUMBER_OF_CARDS_PER_SUIT + 1
    action_index = (n_cards_index * (C.NUMBER_OF_CARDS_PER_SUIT + 1) + (internal_card_number -1))

    # Action index in env.step is 1-based, tensor is 0-based
    # Check if the calculated index (0-based) is valid in the tensor
    if action_index < len(action_space_tensor) and action_space_tensor[action_index] > 0:
         print(f"[Action Conversion] Found valid action index: {action_index + 1} for {n_cards}x card {card_value}")
         return action_index + 1 # Return 1-based index for env.step
    else:
         print(f"[Action Conversion] Error: Move {n_cards} x card {card_value} (index {action_index}) not found/valid in action space.")
         # print(f"Action space tensor (first 60): {action_space_tensor[:60]}") # Debug if needed
         return -1

def get_pass_action_index(env: ScumEnv) -> int:
    """Gets the action index corresponding to 'pass'."""
    # Pass action is the last action, index = total actions
    pass_index_1_based = C.NUMBER_OF_SUITS * (C.NUMBER_OF_CARDS_PER_SUIT + 1) + 1
    pass_index_0_based = pass_index_1_based - 1

    action_space_tensor = env.get_action_space()
    if pass_index_0_based < len(action_space_tensor) and action_space_tensor[pass_index_0_based] > 0:
        print(f"[Action Conversion] Found valid pass action index: {pass_index_1_based}")
        return pass_index_1_based # Return 1-based index
    else:
        print(f"[Action Conversion] Error: Pass action (index {pass_index_0_based}) not found/valid in action space.")
        return -1

# --- API Endpoints ---

@app.route('/game/start', methods=['POST'])
def start_game():
    global game_env, agent_pool
    print("\n--- Received request to /game/start ---")

    try:
        # Initialize Environment
        num_players = C.NUMBER_OF_AGENTS
        game_env = ScumEnv(number_players=num_players)
        game_env.reset()
        agent_pool = AgentPool(num_agents=num_players)
        agent_pool = agent_pool.create_agents_with_paths({'model_id': 'testing', 'model_tag': 'testing', 'model_size': 'small'})

        print(f"Game environment initialized for {num_players} players.")
        print(f"Initial hands (internal): {game_env.cards}")
        print(f"Starting player: {game_env.player_turn}")

        # --- Run ONLY the FIRST AI turn if AI starts ---
        # NO LOOP HERE ANYMORE
        if game_env.player_turn != 0 and sum(game_env.players_in_game) > 1:
            print(f"\n--- First Turn is AI (Player {game_env.player_turn}) ---")
            if game_env.players_in_round[game_env.player_turn]:
                agent = agent_pool.get_agent(game_env.player_turn)
                action_space = game_env.get_action_space()
                state = game_env.get_state(action_space)

                if state is not None and action_space is not None:
                    action, _ = agent.decide_move(state, action_space)
                    action = int(action.item())

                    env_utils.decode_action(action)

                    _, _, done, _ = game_env.step(action, state)
                else:
                     print("Warning: State or action space is None during initial AI turn.")
            else:
                 print(f"AI Player {game_env.player_turn} is not in round initially? Skipping turn logic.")
                 # This case is unlikely on start, but if it happens, env state is returned as is.
                 # The frontend will then trigger the next AI turn if needed.

        # --- Prepare and send initial state ---
        initial_state = get_game_state_for_ui(game_env, message="Game started.")
        print(f"--- Sending initial state to UI (Player {initial_state.get('currentPlayerIndex')}'s turn) ---")
        return jsonify(initial_state)

    except Exception as e:
        print(f"Error during /game/start: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/game/action', methods=['POST'])
def handle_action():
    global game_env, agent_pool
    print("\n--- Received request to /game/action ---")

    if not game_env or not agent_pool:
        print("Error: Game not initialized.")
        return jsonify({"error": "Game not initialized"}), 400

    data = request.get_json()
    action_type = data.get('action_type')
    print(f"Received action type: {action_type}")

    try:
        current_player_turn = game_env.player_turn
        action_description = ""
        action_to_execute = -1
        is_ai_turn_request = (action_type == 'ai_turn')

        print("Game player turn", game_env.player_turn)
        print("Game last player turn", game_env.last_player)

        # --- Process Action based on Type ---

        if is_ai_turn_request:
            # --- AI Turn Request ---
            if current_player_turn == 0:
                print("Error: Received AI turn request, but it's human's turn.")
                return jsonify({"error": "Mismatch: Expected AI turn, but game state indicates human turn."}), 400 # Bad request

            print(f"\n--- Processing AI Turn (Player {current_player_turn}) ---")
            if not game_env.players_in_round[current_player_turn]:
                 print(f"AI Player {current_player_turn} is not in round, skipping turn logic.")
                 # If AI is not in round, we still need to advance the turn
                 game_env._update_player_turn()
                 print(f"Skipped. Next player is now {game_env.player_turn}")
                 final_state_ui = get_game_state_for_ui(game_env, message=f"Player {current_player_turn} passed (not in round).")
                 return jsonify(final_state_ui)

            agent = agent_pool.get_agent(current_player_turn)
            action_space = game_env.get_action_space()
            state = game_env.get_state(action_space)

            if state is None or action_space is None:
                 print(f"Warning: State or action space is None during AI turn {current_player_turn}.")
                 # Return current state with error? Or let it proceed? Let's return error state.
                 return jsonify({"error": f"Internal state error for AI {current_player_turn}"}), 500

            action_to_execute, _ = agent.decide_move(state, action_space)
            action_description = f"AI Player {current_player_turn} chose action {action_to_execute}"
            env_utils.decode_action(action_to_execute) # Log the decoded action

        elif action_type == 'play' or action_type == 'pass':
            # --- Human Turn Request ---
            if current_player_turn != 0:
                print(f"Error: Received human action '{action_type}', but it's Player {current_player_turn}'s turn.")
                return jsonify({"error": f"It's not your turn (Player {current_player_turn}'s turn)."}), 400

            if action_type == 'play':
                cards_played_ui = data.get('cards', [])
                if not cards_played_ui:
                     return jsonify({"error": "No cards provided for play action"}), 400
                action_description = f"You played {len(cards_played_ui)}x {cards_played_ui[0]['cardFace']}"
                print(f"Attempting to process play action: {action_description}")
                action_to_execute = find_action_index(game_env, cards_played_ui)
                if action_to_execute == -1:
                    print("Invalid play action determined.")
                    current_state_ui = get_game_state_for_ui(game_env, message="Invalid move. Try again.")
                    return jsonify(current_state_ui)

            elif action_type == 'pass':
                action_description = "You passed"
                print(f"Attempting to process pass action.")
                action_to_execute = get_pass_action_index(game_env)
                if action_to_execute == -1:
                     print("Invalid pass action determined (not available?).")
                     current_state_ui = get_game_state_for_ui(game_env, message="Cannot pass right now.")
                     return jsonify(current_state_ui)
        else:
            return jsonify({"error": f"Invalid action_type: {action_type}"}), 400

        # --- Step Environment with the Determined Action ---
        if action_to_execute != -1:
            print(f"Executing action index: {action_to_execute} ({action_description})")
            action_space = game_env.get_action_space() # Get action space *before* step
            state = game_env.get_state(action_space)   # Get state *before* step

            if state is None or action_space is None:
                 print(f"Error: State or action space is None before step for player {current_player_turn}.")
                 return jsonify({"error": "Internal state error before step"}), 500

            _, _, player_done, _ = game_env.step(action_to_execute, state)

            print(f"  After step Actual Player: {game_env.player_turn}")
            print(f"  After step Last player: {game_env.last_player}")


        else:
             # Should have been handled earlier (invalid human move, AI skip)
             print("Warning: Reached end of action handler with action_to_execute == -1")

        # --- Prepare and send updated state ---
        final_state_ui = get_game_state_for_ui(game_env) # Phase/message determined internally
        print(f"--- Sending updated state to UI (Player {final_state_ui.get('currentPlayerIndex')}'s turn) ---")
        return jsonify(final_state_ui)

    except Exception as e:
        print(f"Error during /game/action: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500


if __name__ == '__main__':
    # Consider using a more production-ready server like gunicorn or waitress
    # For development, Flask's built-in server is fine
    app.run(debug=True, port=5000) # debug=True enables auto-reloading