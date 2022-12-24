# Slab Params:
BETA_DIST_ALPHA = 4
BETA_DIST_BETA = 5
th_slab_max = 15.5
th_slab_min = 2.1

# Furnace Params:
n_furnaces = 2
FUR1_EFF = 0.95
FUR2_EFF = 0.05

# Simulation Params:
SKIP_TRAINING = False
USE_PREV_MODEL = False
TOTAL_STEPS = 1  # steps per episode
DISCOUNT_FACTOR = 0
NUM_EPISODES = 100000  # Total episodes
EPSILON = 0.05
LR = 0.01

# Other Params:
REWARD_NRMLZ_CONSTANT = 500  # minutes
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
