# server.py
import sys
import os

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(project_root)
# --- End Path Addition ---

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import uuid # Import uuid to generate unique game IDs
import threading # Import threading for locking access to shared state
import pymongo # Import pymongo
from typing import List

from agent.agent_pool import AgentPool
from environment.gymnasium_env import ScumEnv
from config.constants import Constants as C
from db.db import MongoDBManager # Import the DB Manager
import numpy as np

from utils import data_utils, env_utils
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


app = Flask(__name__)
CORS(app)

# --- Global Game State Management ---
active_games = {} 
game_lock = threading.Lock() # Lock to ensure thread-safe access to active_games

# --- Database Setup ---
try:
    # Configure your MongoDB connection here
    db_manager = MongoDBManager(database=C.DB_UI)
    # Ensure unique index on player name
    db_manager.create_index('players', 'name', unique=True)
    print(f"Connected to MongoDB database '{C.DB_UI}' and ensured unique index on 'players.name'.")
except Exception as e:
    print(f"FATAL: Could not connect to MongoDB or create index. {e}")
    raise ValueError

# --- Card Mapping (Keep as is) ---
FACE_MAP_TO_UI = {1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q', 11: 'K', 12: 'A', 13: '2', 14: '2'}
SUITS = ['Spade', 'Club', 'Diamond', 'Heart']

def map_hand_to_ui(hand_values: list) -> list:
    """Maps a list of internal card values to UI card objects for a player's hand."""
    ui_hand = []
    # Sort by value for consistent display in hand
    for i, value in enumerate(sorted(hand_values)):
        face = FACE_MAP_TO_UI.get(value, '?')
        suit = SUITS[i % len(SUITS)] # Cycle through suits
        if value == 13 and (suit == 'Heart'):
            suit = 'Spade' # Keep special suits if desired
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
        suit = SUITS[i % len(SUITS)] # Cycle through suits
        if value == 13 and (suit == 'Heart'):
            suit = 'Spade' # Keep special suits if desired
        if value == 14: suit = 'Heart' # Keep special suits if desired

        ui_cards.append({
            "id": f"table-{face}-{suit}-{i}-{random.random()}", # Unique ID for table cards
            "cardFace": face,
            "suit": suit,
            "value": value
        })
    return ui_cards

# --- get_game_state_for_ui Modification ---
def get_game_state_for_ui(env: ScumEnv, player_names: list, message: str = "", phase: str = None) -> dict:
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

        # Use the provided player names
        player_name = player_names[i] if i < len(player_names) else f"Player {i}"

        players_ui.append({
            "id": f"player-{i}",
            "name": player_name, # Use the name from the list
            "hand": ui_hand,
            "cardCount": len(raw_hand),
            "isHuman": i == 0,
            "finishedRank": env.player_order.index(i) + 1 if i in env.player_order else None,
            "isStillInRound": is_still_in_round,
        })

    # Determine game phase if not explicitly provided
    current_phase = phase
    if current_phase is None:
        if sum(env.players_in_game) <= 1:
            current_phase = 'roundOver'
        else:
            current_phase = 'playing'

    # Determine game message
    current_message = message
    if not message:
        if current_phase == 'playing':
            if 0 <= env.player_turn < len(players_ui):
                 current_message = f"{players_ui[env.player_turn]['name']}'s turn."
                 if not env.players_in_round[env.player_turn]:
                     current_message += " (Must Pass)"
            else:
                 current_message = "Waiting for next turn..."
        elif current_phase == 'roundOver':
            winner_idx = env.player_order[0] if env.player_order and len(env.player_order) > 0 and env.player_order[0] != -1 else -1
            winner_name = players_ui[winner_idx]['name'] if winner_idx != -1 else "Someone"
            current_message = f"Round Over! {winner_name} finished first!"

    current_pile_ui = []
    if hasattr(env, 'current_pile'):
        current_pile_ui = map_pile_to_ui(env.current_pile)

    return {
        "gamePhase": current_phase,
        "players": players_ui, # Already contains the correct names
        "cardsOnTable": current_pile_ui,
        "currentPlayerIndex": env.player_turn if current_phase == 'playing' else -1,
        "gameMessage": current_message,
        "humanPlayerId": "player-0",
    }

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
    if card_value == 14:
        action_index = _handle_two_of_hearts(env)
    else:
        if internal_card_number == 0:
            internal_card_number = C.NUMBER_OF_CARDS_PER_SUIT + 1
        action_index = (n_cards_index * (C.NUMBER_OF_CARDS_PER_SUIT + 1) + (internal_card_number -1))

    if action_index < len(action_space_tensor) and action_space_tensor[action_index] > 0:
         return action_index + 1 # Return 1-based index for env.step
    else:
         print(f"[Action Conversion] Error: Move {n_cards} x card {card_value} (index {action_index}) not found/valid in action space.")
         return -1

def _handle_two_of_hearts(env: ScumEnv) -> int:
    action_space_tensor = env.get_action_space()
    indices = torch.where(action_space_tensor == 1)[0]

    print("Action space tensor: ", action_space_tensor)
    print("Indices: ", indices)

    action_index = indices[torch.where(indices % 14 == 13)[0].item()]
    if len(indices) > 0:
        return action_index
    return -1

def get_pass_action_index(env: ScumEnv) -> int:
    """Gets the action index corresponding to 'pass'."""
    # Pass action is the last action, index = total actions
    pass_index_1_based = C.NUMBER_OF_SUITS * (C.NUMBER_OF_CARDS_PER_SUIT + 1) + 1
    pass_index_0_based = pass_index_1_based - 1

    action_space_tensor = env.get_action_space()
    if pass_index_0_based < len(action_space_tensor) and action_space_tensor[pass_index_0_based] > 0:
        return pass_index_1_based # Return 1-based index
    else:
        print(f"[Action Conversion] Error: Pass action (index {pass_index_0_based}) not found/valid in action space.")
        return -1

# --- Normalization Helper ---
def normalize_name(name: str) -> str:
    return name.lower().strip()

# --- API Endpoints ---

# --- /game/start Modification ---
@app.route('/game/start', methods=['POST'])
def start_game():
    print("\n--- Received request to /game/start ---")
    data = request.get_json()
    if not data or 'playerName' not in data:
        return jsonify({"error": "Missing playerName in request"}), 400

    raw_player_name = data.get('playerName', 'Player 0') # Default just in case
    if not raw_player_name.strip():
         return jsonify({"error": "Player name cannot be empty"}), 400

    human_player_name_normalized = normalize_name(raw_player_name)

    try:
        # --- Ensure Player Exists in DB ---
        if db_manager:
            try:
                db_manager.update_one(
                    'players',
                    {'name': human_player_name_normalized},
                    {'$setOnInsert': {'name': human_player_name_normalized, 'wins': 0, 'plays': 0}},
                    upsert=True
                )
                print(f"Ensured player '{human_player_name_normalized}' exists in DB.")
            except Exception as db_error:
                print(f"Warning: DB error ensuring player exists: {db_error}")
                # Decide if you want to proceed without DB record or return error

        # Generate a unique ID for this new game
        game_id = str(uuid.uuid4())

        # Initialize Environment and Agent Pool
        num_players = C.NUMBER_OF_AGENTS
        local_env = ScumEnv(number_players=num_players)
        local_env.reset()
        local_env.player_turn = np.random.randint(0, 4)
        local_pool = AgentPool(num_agents=num_players)
        # Load your agent models as before...
        local_pool = local_pool.create_agents_with_parameters(
             {
                'model_id': '4efbec94-219a-4f7b-8805-8be90487019f',
                'model_tag': 'testing',
                'model_size': 'large-sep-arch',
                'load_model_path': './models/model_4efbec94-219a-4f7b-8805-8be90487019f_200000.pt'
                }
        )

        # --- Create Player Name List ---
        player_names = [human_player_name_normalized] + [f"Player {i}" for i in range(1, num_players)]

        print(f"Game [{game_id}] environment initialized for {num_players} players.")
        print(f"Game [{game_id}] Player names: {player_names}")
        print(f"Game [{game_id}] Starting player index: {local_env.player_turn}")

        # --- Run ONLY the FIRST AI turn if AI starts ---
        if local_env.player_turn != 0 and sum(local_env.players_in_game) > 1:
            print(f"\n--- Game [{game_id}] First Turn is AI (Player {local_env.player_turn}) ---")
            if local_env.players_in_round[local_env.player_turn]:
                agent = local_pool.get_agent(local_env.player_turn)
                action_space = local_env.get_action_space()
                state = local_env.get_state(action_space)

                if state is not None and action_space is not None:
                    action, _ = agent.decide_move(state, action_space)
                    # env_utils.decode_action(action) # Log decoded action
                    _, _, done, _ = local_env.step(action, state)
                    print(f"  Game [{game_id}] AI Player {local_env.last_player} took first turn. Next is Player {local_env.player_turn}")
                else:
                     print(f"Warning: Game [{game_id}] State or action space is None during initial AI turn.")
            else:
                 print(f"Game [{game_id}] AI Player {local_env.player_turn} is not in round initially? Skipping turn logic.")

        # --- Store the new game instance (including player names) ---
        with game_lock:
            active_games[game_id] = {
                'env': local_env,
                'pool': local_pool,
                'player_names': player_names # Store names with the game
            }
        print(f"Game [{game_id}] added to active games. Total active: {len(active_games)}")

        # --- Prepare and send initial state (using player_names) ---
        initial_state = get_game_state_for_ui(local_env, player_names, message="Game started.")
        initial_state['gameId'] = game_id # Add game_id to the response
        return jsonify(initial_state)

    except Exception as e:
        print(f"Error during /game/start: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

# --- /game/action Modification ---
@app.route('/game/action', methods=['POST'])
def handle_action():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    game_id = data.get('gameId')
    action_type = data.get('action_type')

    if not game_id:
        print("Error: Missing gameId in request.")
        return jsonify({"error": "Missing gameId"}), 400

    # --- Retrieve the specific game instance ---
    game_data = None
    with game_lock:
        game_data = active_games.get(game_id)

    if not game_data:
        print(f"Error: Game ID {game_id} not found in active games.")
        return jsonify({"error": "Game not found or already finished"}), 404

    local_env: ScumEnv = game_data['env']
    local_pool: AgentPool = game_data['pool']
    player_names: List[str] = game_data['player_names'] # Retrieve player names

    try:
        current_player_turn = local_env.player_turn
        action_to_execute = -1
        is_ai_turn_request = (action_type == 'ai_turn')

        if is_ai_turn_request:
            if current_player_turn == 0:
                print(f"Error: Game [{game_id}] Received AI turn request, but it's human's turn.")
                return jsonify({"error": "Mismatch: Expected AI turn, but game state indicates human turn."}), 400

            ai_player_name = player_names[current_player_turn]
            if not local_env.players_in_round[current_player_turn]:
                 local_env._update_player_turn() # Advance turn
                 final_state_ui = get_game_state_for_ui(local_env, player_names, message=f"{ai_player_name} passed (not in round).")
                 return jsonify(final_state_ui)

            agent = local_pool.get_agent(current_player_turn)
            action_space = local_env.get_action_space()
            state = local_env.get_state(action_space)

            if state is None or action_space is None:
                 return jsonify({"error": f"Internal state error for AI {current_player_turn}"}), 500

            action_to_execute, _ = agent.decide_move(state, action_space)

        elif action_type == 'play' or action_type == 'pass':
            if current_player_turn != 0:
                ai_player_name = player_names[current_player_turn]
                print(f"Error: Game [{game_id}] Received human action '{action_type}', but it's {ai_player_name}'s turn.")
                return jsonify({"error": f"It's not your turn ({ai_player_name}'s turn)."}), 400

            if action_type == 'play':
                cards_played_ui = data.get('cards', [])
                if not cards_played_ui:
                     return jsonify({"error": "No cards provided for play action"}), 400
                action_to_execute = find_action_index(local_env, cards_played_ui)
                if action_to_execute == -1:
                    current_state_ui = get_game_state_for_ui(local_env, player_names, message="Invalid move. Try again.")
                    return jsonify(current_state_ui)

            elif action_type == 'pass':
                action_to_execute = get_pass_action_index(local_env)
                if action_to_execute == -1:
                     current_state_ui = get_game_state_for_ui(local_env, player_names, message="Cannot pass right now.")
                     return jsonify(current_state_ui)
        else:
            return jsonify({"error": f"Invalid action_type: {action_type}"}), 400

        # --- Step Environment ---
        if action_to_execute != -1:
            action_space = local_env.get_action_space()
            state = local_env.get_state(action_space)

            if state is None or action_space is None:
                 # print(f"Error: Game [{game_id}] State or action space is None before step for player {current_player_turn}.")
                 return jsonify({"error": "Internal state error before step"}), 500

            local_env.step(action_to_execute, state)

        else:
            print(f"Warning: Game [{game_id}] Reached end of action handler with action_to_execute == -1")

        # --- Prepare and send updated state (using player_names) ---
        final_state_ui = get_game_state_for_ui(local_env, player_names)

        # --- Check for Round/Game End and Update Stats ---
        game_phase = final_state_ui.get("gamePhase")
        if game_phase == 'roundOver' or game_phase == 'gameOver':
            if db_manager:
                try:
                    winner_name = None
                    ranked_players = sorted(final_state_ui.get("players", []), key=lambda p: p.get('finishedRank') or 999)

                    for player_info in ranked_players:
                        p_name = player_info['name'] # Already normalized for human
                        p_rank = player_info.get('finishedRank')

                        db_manager.update_one(
                            'players',
                            {'name': p_name},
                            {'$inc': {'plays': 1}},
                            upsert=True 
                        )
                        if p_rank == 1:
                            winner_name = p_name
                            db_manager.update_one(
                                'players',
                                {'name': winner_name},
                                {'$inc': {'wins': 1}}
                            )
                            # print(f"  Incremented wins for winner: {winner_name}")

                except Exception as db_error:
                    print(f"Warning: DB error updating stats for game {game_id}: {db_error}")
            else:
                pass

            # --- Cleanup finished game ---
            with game_lock:
                 if game_id in active_games:
                     del active_games[game_id]

        print(f"--- Game [{game_id}] Sending updated state to UI (Player index {final_state_ui.get('currentPlayerIndex')}'s turn) ---")
        return jsonify(final_state_ui)

    except Exception as e:
        print(f"Error during /game/action for game {game_id}: {e}")
        import traceback
        traceback.print_exc()
        with game_lock:
            if game_id in active_games:
                del active_games[game_id]
                # print(f"Game [{game_id}] removed due to error. Total active: {len(active_games)}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

# --- NEW: Leaderboard Endpoint ---
@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    #   print("\n--- Received request to /leaderboard ---")
    if not db_manager:
        return jsonify({"error": "Database connection not available"}), 503 # Service Unavailable

    try:
        leaderboard_data = db_manager.find_many(
            collection='players',
            query={
                'wins': {'$gt':-1},  # Optionally only show players with wins
                'name': {'$nin': ['Player 1', 'Player 2', 'Player 3', 'Player 4']} # Exclude specific players
            },
            projection={'name': 1, 'wins': 1, '_id': 0},  # Get name and wins, exclude ID
            sort=[('wins', pymongo.DESCENDING)],  # Sort by wins descending
            limit=10  # Limit to top 20 players
        )
        # print(f"Fetched {len(leaderboard_data)} players for leaderboard.")
        return jsonify(leaderboard_data)

    except Exception as e:
        print(f"Error fetching leaderboard: {e}")
        return jsonify({"error": f"Internal server error fetching leaderboard: {e}"}), 500

# --- NEW: Model Opinion Endpoint ---
@app.route('/game/model-opinion', methods=['POST'])
def get_model_opinion_api():
    """API endpoint to get the AI model's opinion about the current game state."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    game_id = data.get('gameId')
    if not game_id:
        return jsonify({"error": "Missing gameId"}), 400

    # Retrieve the specific game instance
    game_data = None
    with game_lock:
        game_data = active_games.get(game_id)

    if not game_data:
        return jsonify({"error": "Game not found or already finished"}), 404

    local_env: ScumEnv = game_data['env']
    local_pool: AgentPool = game_data['pool']

    try:
        # Get model opinion
        opinion_data = get_model_opinion(local_env, local_pool)
        return jsonify(opinion_data)
    except Exception as e:
        print(f"Error getting model opinion for game {game_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

def get_model_opinion(local_env: ScumEnv, local_pool: AgentPool) -> dict:
    """Gets the AI model's opinion about the current game state and returns it as a dictionary."""
    agent = local_pool.get_agent(1)  # Use agent 1 for consistency
    action_space = local_env.get_action_space()
    state = local_env.get_state(action_space)
    
    if state is None or action_space is None:
        return {"error": "Cannot analyze current game state"}
    
    compact_state = data_utils.compact_form_of_states(state)
    compact_state = compact_state.to(device=C.DEVICE)
    value_prediction, probs_prediction = agent.predict_value_and_probs(compact_state)

    action_space = action_space.unsqueeze(0)
    masked_predictions = data_utils.mask_impossible_actions(action_space, probs_prediction)
    masked_predictions_prob = F.softmax(masked_predictions, dim=-1)

    prediction_masked = Categorical(masked_predictions_prob)
    action = prediction_masked.sample()
    
    # Extract the scalar value from the action tensor and convert to Python int
    action_item = int(action.item())
    probability = float(prediction_masked.probs[0, action_item].item())
    
    # Get decoded action information
    decoded_action = env_utils.decode_action_for_api(action_item + 1)  # +1 because env expects 1-based
    print("Decoded action: ", decoded_action)
    
    # Convert value prediction to scalar
    value_pred_scalar = float(value_prediction.item() if hasattr(value_prediction, 'item') else value_prediction)
    
    # Get top 3 actions with probabilities
    top_actions = []
    probs = masked_predictions_prob[0].cpu().detach().numpy()
    top_indices = probs.argsort()[-3:][::-1]  # Get indices of top 3 values
    
    for idx in top_indices:
        if probs[idx] > 0.001:  # Only include actions with non-negligible probability
            # Convert numpy types to Python native types
            top_actions.append({
                "action": env_utils.decode_action_for_api(int(idx) + 1),
                "probability": float(probs[idx])
            })
    
    return {
        "recommendedAction": decoded_action,
        "probability": probability,
        "expectedValue": value_pred_scalar,
        "topActions": top_actions,
        "confidence": "high" if probability > 0.7 else "medium" if probability > 0.4 else "low"
    }

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000, threaded=True)