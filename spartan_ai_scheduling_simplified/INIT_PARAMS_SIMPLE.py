# Waiting-bay Params
WAIT_BAY_LEN = 10
MIN_CAPACITY_WB = 9

# Slab Params:
BETA_DIST_ALPHA = 4
BETA_DIST_BETA = 5
th_slab_max = 15.5
th_slab_min = 2.1
TIME_CONSTANT = 8 * 60

# Furnace Params:
n_furnaces = 2
FURNACE_EFFICIENCY = [0.95, 0.9, 0.01, 0.95, 0.9]
FURNACE_HEIGHT = [5, 5, 5, 5, 5]
FURNACE_WIDTH = [5, 5, 5, 5, 5]
HEAT_CURVE_TYPE = [1, 0, 1, 1, 0]
# 1 - Sinusoidal, 0 - inverse sinusoidal

# Temporal Params:
UNIT_TIME_STEP = 1

# State Params:

# Action Params:

# Reward Params:
reward_improper_pick_wb = -1  # Breaks the episode. Hence, discourage.
reward_proper_pick_wb = +0.1

reward_improper_remove_fur = -0.5 # Try to pick slab from empty furnace
reward_proper_remove_fur = +0.1
threshold_rolling_time = 2    # +/-2 mins close to rolling finishes
reward_rolling_block = -0.25  # Removing slab while other slab is being rolled

threshold_overcook = 10      # +/-10 mins of over or undercook is acceptable
reward_propercook = +0.5
reward_overcook = -0.2
reward_undercook = -0.2

reward_improper_add_fur = -0.5 # Try to add slab to full furnace
reward_proper_add_fur = +0.1

reward_overcooks_in_furnace = -0.005 # Say 30 mins average overcook - 30 * 0.001 = 0.03 per step
reward_longer_heating_in_furnace = -0.001 # Average heating time 200 mins - 200 *0.001 = -0.2

reward_per_step = -0.01

# Simulation Params:
# State Params:
N_MAPS_WB = 2
N_MAPS_FUR = 2
N_BATCHES = 10

L_REPLAY_BUFFER = 1000
SKIP_TRAINING = False
USE_PREV_MODEL = False
MAX_TIME_STEPS = 100  # steps per episode
TOTAL_PLACED = 100    # Slabs - total placed in furnace per episode
TOTAL_REMOVED = 100   # Slabs - total removed from furnace per episode
DISCOUNT_FACTOR = 0.9
NUM_EPISODES = 100000  # Total episodes
EPSILON = 0.05        # Exploration rate - Not used for PG models
LR_CRITIC = 0.01
LR_ACTOR = 0.01

# Other Params:
HEATING_TIME_NRMLZ = 500  # minutes
SEED = 42

# Testing Params:
TEST_SAMPLES = 5000

# Report saving:
EPISODE_SAVE_TRIGGER = 5000  # Episodes
MODEL = './reports/spartan_ai_simple_model'
LOG_FILE = 'logs.txt'
DATA_SAVE_NAME = './reports/simplified_model_test_data.csv'
TEST_RECORDS = './reports/compiled_data.csv'
EPOCH_VS_REWARD_FIG = './reports/Training_RewardsVsEpisodes.jpg'
