# run_lstm_sac_seed42.py
import config
config.RANDOM_SEED = 42
config.TASCAR_MODEL_PATH = "trained_model_lstm_sac_seed42/"
config.TASCAR_RESULTS    = "results_lstm_sac_seed42/"
import runpy
runpy.run_path("train_lstm_sac.py", run_name="__main__")