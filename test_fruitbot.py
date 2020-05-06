###
# Testing "generalization" code for Proc-gen Fruitbot "easy" environment
# Kaahan Radia, Omkar Shanbhag, Aniket Mandalik
###

import tensorflow as tf
import numpy as np
from procgen import ProcgenEnv
import argparse
import os
from datetime import datetime

from baselines.ppo2 import ppo2
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
    # get model path
    parser = argparse.ArgumentParser(description="Parse testing arguments")
    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help='Path to model checkpoint.')
    parser.add_argument('--config',
                        type=str,
                        default='configurations/ppo_baseline_cuda.yaml',
                        help='Path to configuration file.')
    args = parser.parse_args()
    if args.model_path is None or not os.path.exists(args.model_path):
        raise OSError("Invalid model file supplied")

    # create configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    # create save directory
    model_file_path = args.model_path
    exp_creation_time = os.path.normpath(model_file_path).split(os.sep)[-3]
    print(exp_creation_time)
    exp_dir = f"runs/{cfg.EXPERIMENT_NAME}/{exp_creation_time}_test/"
    os.makedirs(exp_dir, exist_ok=True)


    # create logger
    format_strs = ['csv', 'stdout']
    logger.configure(dir=exp_dir, format_strs=format_strs, log_suffix=datetime.now().strftime('%Y-%m-%d-%H-%M'))

    # create (vectorized) procgen environment
    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=cfg.TEST.NUM_ENVS,
                      env_name="fruitbot",
                      num_levels=cfg.TEST.NUM_LEVELS,
                      start_level=cfg.TEST.LEVEL_SEED,
                      distribution_mode="easy")
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )
    venv = VecNormalize(venv=venv, ob=False)

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
    logger.info("testing")
    ppo2.learn(
        env=venv,
        network=conv_fn,
        total_timesteps=cfg.TRAIN.TOTAL_TIMESTEPS,
        save_interval=0,
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
        test=True,
        load_path=model_file_path
    )


if __name__ == '__main__':
    main()
