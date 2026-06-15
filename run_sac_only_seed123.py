# run_sac_only_seed123.py
# Runs train_sac_only.py with seed=123 and a
# separate output path, WITHOUT modifying
# train_sac_only.py or config.py.

import config
config.RANDOM_SEED = 123
config.SAC_ONLY_MODEL_PATH = "trained_model_sac_only_seed123/"

import runpy
runpy.run_path("train_sac_only.py", run_name="__main__")