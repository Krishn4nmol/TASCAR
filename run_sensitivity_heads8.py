import config
config.RANDOM_SEED = 42
config.TRANSFORMER_HEADS = 8
config.TASCAR_MODEL_PATH = "trained_model_sensitivity_heads8/"
config.TASCAR_RESULTS    = "results_sensitivity_heads8/"
import runpy
runpy.run_path("train_tascar.py", run_name="__main__")