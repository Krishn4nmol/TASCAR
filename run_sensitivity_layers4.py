import config
config.RANDOM_SEED = 42
config.TRANSFORMER_LAYERS = 4
config.TASCAR_MODEL_PATH = "trained_model_sensitivity_layers4/"
config.TASCAR_RESULTS    = "results_sensitivity_layers4/"
import runpy
runpy.run_path("train_tascar.py", run_name="__main__")