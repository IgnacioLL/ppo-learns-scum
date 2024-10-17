import torch

class Constants:
    NUMBER_OF_SUITS = 4
    NUMBER_OF_CARDS_PER_SUIT = 13
    NUMBER_OF_CARDS = NUMBER_OF_SUITS * NUMBER_OF_CARDS_PER_SUIT

    NUMBER_OF_AGENTS = 5

    N_CARDS_TO_TEXT = {0:'a', 1: 'a pair of', 2: 'a triplet of', 3: 'a quad of', 4: 'PASS!'}

    NUMBER_OF_POSSIBLE_STATES = (NUMBER_OF_CARDS_PER_SUIT+1)*NUMBER_OF_SUITS + 1

    REPLAY_MEMORY_SIZE = 100_000 ## 
    MIN_REPLAY_MEMORY_SIZE = 1_000

    BATCH_SIZE = 64
    DISCOUNT = .99
    UPDATE_TARGET_EVERY = 1_000

    EPISODES = 100_000

    REWARD_PASS = -0.1
    REWARD_CARD = 0.05

    REWARD_WIN = 1
    REWARD_SECOND = 0.5
    REWARD_THIRD = 0
    REWARD_FOURTH = -0.5
    REWARD_LOSE = -1
    REWARD_CHOOSE_IMPOSIBLE_ACTION = 0

    REWARD_EMPTY_HAND = 0.2

    EPSILON = 1
    EPSILON_DECAY = 0.999
    MIN_EPSILON = 0.01

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    AGGREGATE_STATS_EVERY = 1_000

    WARMUP_STEPS = 25_000
    INITIAL_LR_FACTOR = 0.01  # Start with 1% of the target learning rate

    CHECKPOINTS_PATH = f"models/checkpoints/"

