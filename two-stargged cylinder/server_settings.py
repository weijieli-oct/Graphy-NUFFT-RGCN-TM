
from pathlib import Path

ROOT = Path(__file__).resolve().parent


ACTION = "prepare"
DEVICE = "cuda:0"


EXPERIMENT = "staggered_cylinders_medium"
RE_LIST = [20., 40., 100., 150.]
AMPLITUDE = 0.0015


TRAINING_OVERRIDES = {}
RUNS_DIRECTORY = ROOT / "runs"
REBUILD_MESH = False
SAVE_MESH_FIGURES = True
PLOT_AFTER_SOLVE = True
STATISTICS_START_TIME = 50.0

RUN_UNSEEN_AFTER_TRAIN = False
UNSEEN_RE = None
SOURCE_CHECKPOINT = r"E:\mesh_cylinder\RGCNTM_Staggered_Cylinders\runs\staggered_cylinders_medium_20261013_121312_145570\train\checkpoint.pt"  # 必须填具体的 checkpoint.pt 文件，不能填文件夹。
RESUME_CHECKPOINT = None
TRANSFER_CONFIG = None


INITIALIZE_CHECKPOINT = None
