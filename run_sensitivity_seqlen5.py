import config
config.RANDOM_SEED = 42
config.SEQUENCE_LENGTH = 5
config.TASCAR_MODEL_PATH = "trained_model_sensitivity_seqlen5/"
config.TASCAR_RESULTS    = "results_sensitivity_seqlen5/"
import runpy
runpy.run_path("train_tascar.py", run_name="__main__")