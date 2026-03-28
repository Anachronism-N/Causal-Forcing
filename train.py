import argparse
import os
import time
from omegaconf import OmegaConf
import wandb

from trainer import DiffusionTrainer, ODETrainer, ScoreDistillationTrainer, ConsistencyDistillationTrainer


def log(message):
    print(
        f"[{time.strftime('%F %T')}] [train.py pid={os.getpid()} rank={os.environ.get('RANK', '?')} local_rank={os.environ.get('LOCAL_RANK', '?')}] {message}",
        flush=True,
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--tf", action="store_true")

    args = parser.parse_args()
    log(f"args parsed: config_path={args.config_path}, logdir={args.logdir}, disable_wandb={args.disable_wandb}, cwd={os.getcwd()}")

    log(f"loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)
    log("loading default config from configs/default_config.yaml")
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    log("config merge complete")
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf 
    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    log(f"trainer type resolved: trainer={config.trainer}, config_name={config_name}")

    if config.trainer == "diffusion":
        log("constructing DiffusionTrainer")
        trainer = DiffusionTrainer(config)
    elif config.trainer == "ode":
        log("constructing ODETrainer")
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        log("constructing ScoreDistillationTrainer")
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "consistency_distillation":
        log("constructing ConsistencyDistillationTrainer")
        trainer = ConsistencyDistillationTrainer(config)
    else:
        raise ValueError(f"Unsupported trainer type: {config.trainer}")

    log("trainer constructed, entering train()")
    trainer.train()
    log("train() finished, calling wandb.finish()")

    wandb.finish()
    log("wandb.finish() done")


if __name__ == "__main__":
    main()
