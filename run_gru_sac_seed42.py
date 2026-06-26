# run_gru_sac_seed42.py
import config
config.RANDOM_SEED = 42
config.TASCAR_MODEL_PATH = "trained_model_gru_sac_seed42/"
config.TASCAR_RESULTS    = "results_gru_sac_seed42/"
import runpy
runpy.run_path("train_gru_sac.py", run_name="__main__")