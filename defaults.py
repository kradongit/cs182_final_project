import os

from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 0
_C.SYSTEM.NUM_WORKS = 0

# Based on PPO2 training from https://github.com/openai/train-procgen/blob/master/train_procgen/train.py
_C.TRAIN = CN()
_C.TRAIN.ALGO = "PPO2"
_C.TRAIN.NETWORK = "NATURE_CNN"
_C.TRAIN.NUM_ENVS = 1
_C.TRAIN.NUM_LEVELS = 50
_C.TRAIN.LEVEL_SEED = 0
_C.TRAIN.LR = 5e-4
_C.TRAIN.GAMMA = .999
_C.TRAIN.LAM = .95
_C.TRAIN.CLIP_RANGE = .2
_C.TRAIN.USE_VF_CLIPPING = True
_C.TRAIN.BATCH_SIZE = 512
_C.TRAIN.MINIBATCHES = 1
_C.TRAIN.NUM_EPOCHS = 3
# total number of actions over all episodes / epochs
_C.TRAIN.TOTAL_TIMESTEPS = 1_000_000
_C.TRAIN.POLICY = ""
_C.TRAIN.AUGMENT = False

# testing defaults
_C.TEST = CN()
_C.TEST.NUM_ENVS = 1
_C.TEST.NUM_LEVELS = 0
_C.TEST.LEVEL_SEED = 0
_C.TEST.TIMESTEPS = 100_000
_C.TEST.BATCH_SIZE = _C.TRAIN.BATCH_SIZE

_C.EXPERIMENT_NAME = "openai_baseline"

# Adapted from https://github.com/rbgirshick/yacs
def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for fruitbot training"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


