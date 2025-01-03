import os
import os.path as osp
from datetime import datetime

from omegaconf import OmegaConf


def load_config(cfg_file):
    cfg_dir = osp.dirname(cfg_file)
    cfg = OmegaConf.load(cfg_file)
    if "_base_" in cfg:
        if isinstance(cfg._base_, str):
            #  base_cfg = OmegaConf.load(osp.join(cfg_dir, cfg._base_))
            base_cfg = load_config(osp.join(cfg_dir, cfg._base_))
        else:
            base_cfg = OmegaConf.merge(*[OmegaConf.load(osp.join(cfg_dir, f)) for f in cfg._base_])
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def get_config(args):
    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, True)

    if args.opts is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))
    if hasattr(args, "batch_size") and args.batch_size:
        cfg.data.batch_size = args.batch_size

    if hasattr(args, "resume") and args.resume:
        cfg.checkpoint.resume = args.resume

    if hasattr(args, "eval") and args.eval:
        cfg.evaluate.eval_only = args.eval

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    total_bs = cfg.data.batch_size * world_size
    cfg.method_name = args.method_name + f"_b{total_bs}"

    now = datetime.now()
    formatted_time = f"{now.year % 100:02d}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}{now.second:02d}"
    if hasattr(args, "output") and args.output:
        cfg.output = args.output + '_' + formatted_time
    else:
        cfg.output = osp.join("output", cfg.method_name + '_' + formatted_time)

    if hasattr(args, "tag") and args.tag:
        cfg.tag = args.tag
        cfg.output = osp.join(cfg.output, cfg.tag)

    if hasattr(args, "wandb") and args.wandb:
        cfg.wandb = args.wandb

    OmegaConf.set_readonly(cfg, True)

    return cfg
