###
# Training code for Proc-gen Fruitbot "easy" environment
# Heavily inspired by https://github.com/openai/train-procgen/blob/master/train_procgen/train.py
# Kaahan Radia, Omkar Shanbhag, Aniket Mandalik
###

import tensorflow as tf
import numpy as np
from procgen import ProcgenEnv
import argparse
import os
from datetime import datetime

from baselines.ppo2 import ppo2
from baselines.a2c import a2c
from baselines.acktr import acktr
from baselines.common.models import build_impala_cnn
from baselines.common.models import nature_cnn
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger

from defaults import get_cfg_defaults


def main():
    parser = argparse.ArgumentParser(description="Process training arguments.")
    parser.add_argument('--config',
                        type=str,
                        default="configurations/ppo_baseline_cuda.yaml",
                        help="config file name (located in config dir)")
    args = parser.parse_args()

    # create configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    print(cfg.TRAIN.TOTAL_TIMESTEPS)

    # create experiment directory
    exp_dir = f"runs/{cfg.EXPERIMENT_NAME}/{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    os.makedirs(exp_dir, exist_ok=True)

    # create logger
    format_strs = ['csv', 'stdout']
    logger.configure(dir=exp_dir, format_strs=format_strs, log_suffix=datetime.now().strftime('%Y-%m-%d-%H-%M'))

    # create (vectorized) procgen environment
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=cfg.TRAIN.NUM_ENVS,
                      env_name="fruitbot",
                      num_levels=cfg.TRAIN.NUM_LEVELS,
                      start_level=cfg.TRAIN.LEVEL_SEED,
                      distribution_mode="easy")
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    venv = VecNormalize(venv=venv, ob=False)

    test_venv = ProcgenEnv(num_envs=cfg.TEST.NUM_ENVS,
                      env_name="fruitbot",
                      num_levels=cfg.TEST.NUM_LEVELS,
                      start_level=cfg.TEST.LEVEL_SEED,
                      distribution_mode="easy")
    test_venv = VecExtractDictObs(test_venv, "rgb")
    test_venv = VecMonitor(
        venv=test_venv, filename=None, keep_buf=100,
    )
    test_venv = VecNormalize(venv=test_venv, ob=False)

    # create tensorflow session
    logger.info("creating tf session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    # create cnn todo: make this less ugly
    conv_fn = None
    logger.info("building cnn")
    if cfg.TRAIN.NETWORK == "NATURE_CNN":
        conv_fn = lambda x: nature_cnn(x)
    elif cfg.TRAIN.NETWORK == "IMPALA_CNN":
        conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    # training
    logger.info("training")
    if cfg.TRAIN.POLICY == "A2C":
        a2c.learn(
            env=venv,
            network=conv_fn,
            total_timesteps=cfg.TRAIN.TOTAL_TIMESTEPS,
            nsteps=cfg.TRAIN.BATCH_SIZE,
            log_interval=1,
            eval_env=test_venv,
            augment=cfg.TRAIN.AUGMENT
        )
    elif cfg.TRAIN.POLICY == "ACKTR":
        acktr.learn(
            env=venv,
            network=conv_fn,
            total_timesteps=cfg.TRAIN.TOTAL_TIMESTEPS,
            nsteps=cfg.TRAIN.BATCH_SIZE,
            log_interval=1,
            eval_env=test_venv,
            augment=cfg.TRAIN.AUGMENT,
            seed=None
        )
    elif cfg.TRAIN.POLICY == "PPO":
        ppo2.learn(
            env=venv,
            eval_env=test_venv,
            network=conv_fn,
            total_timesteps=cfg.TRAIN.TOTAL_TIMESTEPS,
            save_interval=5,
            nsteps=cfg.TRAIN.BATCH_SIZE,
            nminibatches=cfg.TRAIN.MINIBATCHES,
            lam=cfg.TRAIN.LAM,
            gamma=cfg.TRAIN.GAMMA,
            noptepochs=cfg.TRAIN.NUM_EPOCHS,
            log_interval=1,
            clip_vf=cfg.TRAIN.USE_VF_CLIPPING,
            lr=cfg.TRAIN.LR,
            cliprange=cfg.TRAIN.CLIP_RANGE,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            augment=cfg.TRAIN.AUGMENT
        )


if __name__ == '__main__':
    main()





