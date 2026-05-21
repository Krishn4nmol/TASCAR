# config.py
# All settings for CASR project

# ─────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────
DATA_PATH  = "data/"
TRAIN_DAYS = [1, 2, 3, 4, 5]
TEST_DAYS  = [6, 7]

# ─────────────────────────────────────────
# S-CACHE SETTINGS
# ─────────────────────────────────────────
NUM_QUEUES             = 3
QUEUE_BOUNDARIES       = [0, 1, 60, float('inf')]

# Higher initial capacity = better results
INITIAL_QUEUE_CAPACITY = [5000, 500, 100]
WINDOW_CACHE_RATIO     = 0.2

# ─────────────────────────────────────────
# SERVER SETTINGS
# ─────────────────────────────────────────
SERVER_MEMORY_MB            = 4096
DEFAULT_CONTAINER_MEMORY_MB = 128

# ─────────────────────────────────────────
# KEY SETTING: NUMBER OF FUNCTIONS
# This must match between train and evaluate
# 2000 = realistic single server simulation
# ─────────────────────────────────────────
NUM_FUNCTIONS = 2000

# ─────────────────────────────────────────
# KEY SETTING: CALLS PER WORKLOAD
# Higher = more accurate but slower
# 100000 = good balance
# ─────────────────────────────────────────
EVAL_CALLS = 100000

# ─────────────────────────────────────────
# REINFORCEMENT LEARNING SETTINGS
# ─────────────────────────────────────────
THETA          = 0.8
DELTA          = 10000
SCALING_FACTOR = 0.25

# ─────────────────────────────────────────
# PPO SETTINGS
# Exact values from paper Table 2
# ─────────────────────────────────────────
LEARNING_RATE_ACTOR  = 0.001
LEARNING_RATE_CRITIC = 0.001
HIDDEN_LAYER_SIZE    = 128
DISCOUNT_FACTOR      = 0.63
GAE_LAMBDA           = 0.95
PPO_CLIP             = 0.2
ENTROPY_COEFF        = 0.01
MINI_BATCH_SIZE      = 20
REPLAY_BUFFER_SIZE   = 1000
EPOCHS_PER_UPDATE    = 10

# ─────────────────────────────────────────
# TRAINING SETTINGS
# ─────────────────────────────────────────
MAX_EPISODES      = 200
CALLS_PER_EPISODE = 100000
MODEL_SAVE_PATH   = "trained_model/"
PRINT_EVERY       = 10

# ─────────────────────────────────────────
# BASELINE SETTINGS
# ─────────────────────────────────────────
FIXED_KEEPALIVE_SECONDS = 600

# ─────────────────────────────────────────
# RESULTS SETTINGS
# ─────────────────────────────────────────
THETA_VALUES_TO_TEST = [0.2, 0.4, 0.6, 0.8]
RESULTS_PATH         = "results/"

# ─────────────────────────────────────────
# COOLING SETTINGS
# Prevents laptop overheating
# ─────────────────────────────────────────
COOLING_BETWEEN_ALGORITHMS = 30   # seconds
COOLING_BETWEEN_WORKLOADS  = 120  # seconds

# ─────────────────────────────────────────
# YOUR OWN EXPERIMENT
# Change NUM_QUEUES to 4 when experimenting
# ─────────────────────────────────────────
YOUR_NUM_QUEUES       = 4
YOUR_QUEUE_BOUNDARIES = [0, 1, 30, 60,
                         float('inf')]