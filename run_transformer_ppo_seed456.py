# run_transformer_ppo_seed456.py
# Runs train_transformer_ppo.py with seed=456 and
# a separate output path, WITHOUT modifying
# train_transformer_ppo.py or config.py.

import config
config.RANDOM_SEED = 456
config.TRANSFORMER_PPO_MODEL_PATH = "trained_model_transformer_ppo_seed456/"

import runpy
runpy.run_path("train_transformer_ppo.py", run_name="__main__")