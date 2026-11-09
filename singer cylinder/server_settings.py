
from pathlib import Path

ROOT = Path(__file__).resolve().parent


ACTION = "predict"
DEVICE = "cuda:0"


EXPERIMENT = "cylinder_medium"


RE_LIST = None
AMPLITUDE = 0.0015


TRAINING_OVERRIDES = {}
RUNS_DIRECTORY = ROOT / "runs"
REBUILD_MESH = False
SAVE_MESH_FIGURES = True


RUN_UNSEEN_AFTER_TRAIN = False
UNSEEN_RE = None


STATISTICS_START_TIME = 50.0


RESUME_CHECKPOINT = None
SOURCE_CHECKPOINT = r"E:\mesh_cylinder\RGCNTM_Windows_CUDA\runs\cylinder_medium_20261009_164430_918265\train\checkpoint.pt"


TRANSFER_CONFIG = None


INITIALIZE_CHECKPOINT = None
