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
TOTAL_STEPS = 1  # minutes
DISCOUNT_FACTOR = 0
NUM_EPISODES = 30000
SKIP_TRAINING = False
SOLVED_SCORE = 400
EPSILON = 0.05
LR = 0.001

# Testing Params:
REWARD_NRMLZ_CONSTANT = 500 # minutes
TEST_SAMPLES = 5000
DATA_SAVE_NAME = 'simplified_model_test_data.csv'
TEST_RECORDS = 'compiled_data.csv'

# Other Params:
SEED = 42

# Log files:
MODEL = 'spartan_ai_simple_model'
LOG_FILE = 'logs.txt'