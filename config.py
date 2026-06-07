# config.py
# All settings for CASR and TASCAR project

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
INITIAL_QUEUE_CAPACITY = [5000, 500, 100]
WINDOW_CACHE_RATIO     = 0.2

# ─────────────────────────────────────────
# SERVER SETTINGS
# ─────────────────────────────────────────
SERVER_MEMORY_MB            = 4096
DEFAULT_CONTAINER_MEMORY_MB = 128

# ─────────────────────────────────────────
# KEY SETTING: NUMBER OF FUNCTIONS
# ─────────────────────────────────────────
NUM_FUNCTIONS = 2000

# ─────────────────────────────────────────
# KEY SETTING: CALLS PER WORKLOAD
# ─────────────────────────────────────────
EVAL_CALLS = 100000

# ─────────────────────────────────────────
# REINFORCEMENT LEARNING SETTINGS
# ─────────────────────────────────────────
THETA          = 0.8
DELTA          = 10000
SCALING_FACTOR = 0.25

# ─────────────────────────────────────────
# PPO SETTINGS (CASR)
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
COOLING_BETWEEN_ALGORITHMS = 30
COOLING_BETWEEN_WORKLOADS  = 120

# ─────────────────────────────────────────
# TASCAR: TRANSFORMER SETTINGS
# ─────────────────────────────────────────
SEQUENCE_LENGTH    = 10
TRANSFORMER_DIM    = 64
TRANSFORMER_HEADS  = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 128
DROPOUT_RATE       = 0.1

# ─────────────────────────────────────────
# TASCAR: SAC SETTINGS
# ─────────────────────────────────────────
SAC_ALPHA            = 0.2
SAC_TAU              = 0.005
SAC_GAMMA            = 0.63
SAC_LR_ACTOR         = 0.0001
SAC_LR_CRITIC        = 0.0001
SAC_LR_ALPHA         = 0.0001
SAC_BUFFER_SIZE      = 100000
SAC_BATCH_SIZE       = 64
SAC_UPDATE_FREQ      = 1
AUTO_ENTROPY         = True
TARGET_ENTROPY       = -1.0
SAC_UPDATES_PER_STEP = 10

# ─────────────────────────────────────────
# TASCAR: DELTA SETTINGS
# Training uses small delta = more steps
# Evaluation uses same as CASR = fair!
# ─────────────────────────────────────────
TASCAR_DELTA      = 1000
TASCAR_EVAL_DELTA = 10000

# ─────────────────────────────────────────
# TASCAR: DYNAMIC REWARD SETTINGS
# ─────────────────────────────────────────
THETA_MIN        = 0.5
THETA_MAX        = 0.9
THETA_ADAPT_RATE = 0.01

# ─────────────────────────────────────────
# TASCAR: TRAINING SETTINGS
# 500 episodes for better learning!
# ─────────────────────────────────────────
TASCAR_EPISODES   = 500
TASCAR_MODEL_PATH = "trained_model_tascar/"
TASCAR_RESULTS    = "results_tascar/"

# ─────────────────────────────────────────
# EVALUATION METRICS SETTINGS
# Professor recommended metrics!
# ─────────────────────────────────────────

# SLA threshold in seconds
# Cold start > 2s = SLA violation!
SLA_THRESHOLD = 2.0

# Carbon intensity kg CO2 per kWh
# Standard UK grid average!
CARBON_INTENSITY = 0.233

# Power consumption per GB per second
# Standard AWS cloud estimate!
POWER_PER_GB = 0.00125

# Burst detection threshold
# Requests per second = burst!
BURST_THRESHOLD = 100

# ─────────────────────────────────────────
# TPI WEIGHTS
# TASCAR Performance Index!
# Must sum to 1.0!
# ─────────────────────────────────────────
TPI_W1 = 0.25  # CSR (most important!)
TPI_W2 = 0.20  # WMT
TPI_W3 = 0.20  # Throughput
TPI_W4 = 0.20  # SVR
TPI_W5 = 0.15  # RUE

# ─────────────────────────────────────────
# RL METRICS SETTINGS
# For convergence detection!
# ─────────────────────────────────────────

# Window to check convergence
CONVERGENCE_WINDOW = 20

# Reward std below this = converged!
CONVERGENCE_THRESHOLD = 0.05

# ─────────────────────────────────────────
# SCALING METRICS SETTINGS
# For elasticity and SA calculation!
# ─────────────────────────────────────────

# Expected demand per queue
# Based on Azure dataset distribution!
EXPECTED_DEMAND = [5000, 500, 100]

# Elasticity window in calls
ELASTICITY_WINDOW = 10000