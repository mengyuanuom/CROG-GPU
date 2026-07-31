import argparse
import datetime
import os
from pathlib import Path
import shutil
import sys
import time
import warnings
from functools import partial

os.environ["WANDB_MODE"] = "offline"

import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
from utils.dataset import OCIDVLGDataset
from engine.crog_engine import train_with_grasp, validate_with_grasp, validate_without_grasp
from model import build_model
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)
from utils.lr_scheduler import rebuild_multistep_scheduler
from utils.npu import build_grad_scaler, device_count, empty_cache, set_device

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def _replace_with_link_or_copy(source, target):
    """Atomically update a checkpoint alias without duplicating data if possible."""
    source = Path(source)
    target = Path(target)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _replace_metric_alias(source, output_dir, prefix, epoch, metric_suffix):
    """Keep one metric-labelled best checkpoint for a moving best series."""
    output_dir = Path(output_dir)
    target = output_dir / (
        f"{prefix}_epoch_{int(epoch):03d}_{metric_suffix}.pth"
    )
    _replace_with_link_or_copy(source, target)
    for previous in output_dir.glob(f"{prefix}_epoch_*.pth"):
        if previous != target:
            previous.unlink()
    return target


def _normalize_checkpoint_keys(state_dict, model):
    """Match plain and DDP checkpoint key prefixes without changing names."""
    target_keys = tuple(model.state_dict().keys())
    source_keys = tuple(state_dict.keys())
    if not source_keys or not target_keys:
        return state_dict
    source_ddp = source_keys[0].startswith("module.")
    target_ddp = target_keys[0].startswith("module.")
    if source_ddp == target_ddp:
        return state_dict
    if source_ddp:
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return {f"module.{key}": value for key, value in state_dict.items()}


def _resolve_timestamped_checkpoint(checkpoint_path):
    """Resolve a stable base run path to the newest timestamped run."""
    requested = Path(checkpoint_path)
    if requested.is_file():
        return requested
    base_dir = requested.parent
    run_directories = list(base_dir.parent.glob(f"{base_dir.name}_*"))
    candidates = [
        run_directory / requested.name
        for run_directory in run_directories
        if (run_directory / requested.name).is_file()
    ]
    legacy_patterns = {
        "best_iou_model.pth": "best_epoch_*_IoU_*.pth",
        "best_jindex_model.pth": "best_epoch_*_J1_*_J5_*.pth",
    }
    metric_pattern = legacy_patterns.get(requested.name)
    if not candidates and metric_pattern:
        candidates = [
            candidate
            for run_directory in run_directories
            for candidate in run_directory.glob(metric_pattern)
            if candidate.is_file()
        ]
    if not candidates:
        return requested
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)


def _load_maplegrasp_stage1(model, checkpoint_path):
    """Initialize official Stage 2 from a completed Stage-1 checkpoint."""
    requested_path = checkpoint_path
    checkpoint_path = _resolve_timestamped_checkpoint(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError(
            "MapleGrasp Stage-1 checkpoint not found at "
            f"{requested_path!r}, including timestamped run directories. "
            "Train the Stage-1 config first or update TRAIN.weight in the "
            "Stage-2 YAML."
        )
    if str(checkpoint_path) != str(requested_path):
        logger.info(
            "=> resolved latest timestamped Stage-1 checkpoint: '{}'",
            checkpoint_path,
        )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _normalize_checkpoint_keys(state_dict, model)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "module.proj.vis_grasp.weight",
        "module.proj.vis_grasp.bias",
        "proj.vis_grasp.weight",
        "proj.vis_grasp.bias",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Stage-1 checkpoint is not compatible with official MapleGrasp "
            f"Stage 2; missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    logger.info(
        "=> initialized MapleGrasp Stage 2 from Stage-1 checkpoint '{}' "
        "(new grasp head: {})",
        checkpoint_path,
        sorted(incompatible.missing_keys),
    )
    del checkpoint
    empty_cache()


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch(reraise=True)
def main():
    args = get_parser()
    evaluation_protocol = str(
        getattr(args, "evaluation_protocol", "crog_legacy")
    ).strip().lower()
    if evaluation_protocol not in {"crog", "crog_legacy", "crog_source"}:
        raise ValueError(
            "train_crog.py preserves the CROG evaluation protocol; "
            "set TEST.evaluation_protocol=crog_legacy."
        )
    args.evaluation_protocol = "crog_legacy"
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.npus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", args.world_size))
    if args.npus_per_node > device_count():
        raise RuntimeError(
            f"torchrun requested {args.npus_per_node} processes, but only "
            f"{device_count()} visible NPUs were found."
        )
    main_worker(args.local_rank, args)


def main_worker(local_rank, args):
    base_exp_name = str(args.exp_name)
    run_timestamp = os.environ.get("CROG_RUN_TIMESTAMP", "").strip()
    if not run_timestamp and args.world_size > 1:
        raise RuntimeError(
            "Multi-NPU training requires one shared CROG_RUN_TIMESTAMP. "
            "Launch through tools/train_8npu.sh or export it before torchrun."
        )
    if not run_timestamp:
        run_timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )[:-3]
    args.base_exp_name = base_exp_name
    args.run_timestamp = run_timestamp
    args.exp_name = f"{base_exp_name}_{run_timestamp}"
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.rank == 0:
        os.makedirs(args.output_dir, exist_ok=False)
    else:
        for _ in range(200):
            if os.path.isdir(args.output_dir):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"Rank 0 did not create run directory: {args.output_dir}"
            )

    # local rank & global rank
    args.npu = local_rank
    args.device = set_device(local_rank)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.rank,
                 filename="train.log",
                 mode="a")
    logger.info("Run output directory: {}", args.output_dir)

    # dist init
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )
    print(
        f"[HCCL] rank={args.rank}/{args.world_size} "
        f"local_rank={args.local_rank} device={args.device}",
        flush=True,
    )
    args.manual_seed = init_random_seed(
        args.manual_seed,
        device=args.device,
        rank=args.rank,
        world_size=args.world_size,
    )
    set_random_seed(args.manual_seed, deterministic=False)

    # wandb
    # if args.rank == 0:
    #     wandb.init(job_type="training",
    #                mode="online",
    #                config=args,
    #                project="CROG",
    #                name=args.exp_name,
    #                tags=[args.dataset, args.clip_pretrain])
    dist.barrier()

    # build model
    model, param_list = build_model(args)
    needs_offset = bool(getattr(model, "supports_offset", False))
    logger.info(
        "Model architecture: {}, offset supervision: {}",
        getattr(args, "architecture", "crog"),
        needs_offset,
    )
    if args.sync_bn:
        logger.warning(
            "SyncBatchNorm is disabled for the Ascend NPU training path. "
            "torch_npu SyncBatchNorm can trigger device-side AIVector/MTE "
            "faults during multi-NPU training; using per-rank BatchNorm."
        )
        args.sync_bn = False
    logger.info(model)
    logger.info(args)

    # build optimizer & lr scheduler
    # Ascend's multi-tensor ForeachAdd kernel can fail for heterogeneous
    # parameter lists (ETRG is a common trigger). Use the stable per-tensor
    # Adam path by default.
    optimizer_foreach = bool(getattr(args, "optimizer_foreach", False))
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay,
                                 foreach=optimizer_foreach)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    configured_base_lrs = tuple(scheduler.base_lrs)
    amp_enabled = bool(getattr(args, "amp", True))
    scaler = build_grad_scaler(enabled=amp_enabled)
    if args.rank == 0:
        logger.info(
            "Precision path: amp={}, scaler={}",
            amp_enabled,
            type(scaler).__name__,
        )

    model = model.to(args.device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.npu],
        output_device=args.npu,
        find_unused_parameters=True,
    )

    unwrapped_model = model.module
    maplegrasp_stage = getattr(unwrapped_model, "maplegrasp_stage", None)
    if args.weight:
        if args.resume:
            raise ValueError(
                "TRAIN.weight initializes MapleGrasp Stage 2 from Stage 1, "
                "whereas TRAIN.resume continues the same stage; set only one."
            )
        if maplegrasp_stage != 2:
            raise ValueError(
                "TRAIN.weight is reserved for MapleGrasp Stage-2 "
                "initialization in this runner."
            )
        _load_maplegrasp_stage1(model, args.weight)

    # build dataset
    if args.batch_size % args.world_size:
        raise ValueError(
            f"Official global train batch size {args.batch_size} must be divisible "
            f"by world size {args.world_size}."
        )
    if args.batch_size_val % args.world_size:
        raise ValueError(
            f"Official global validation batch size {args.batch_size_val} must be "
            f"divisible by world size {args.world_size}."
        )
    args.global_batch_size = args.batch_size
    args.global_batch_size_val = args.batch_size_val
    args.batch_size = int(args.batch_size / args.world_size)
    args.batch_size_val = int(args.batch_size_val / args.world_size)
    args.workers = int(
        (args.workers + args.world_size - 1) / args.world_size)


    train_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='train',
                            with_depth=bool(getattr(args, "with_depth", False)),
                            version=args.version,
                            with_grasp_offset=needs_offset,
                            offset_r=float(getattr(args, "offset_r", 20.0)),
                            offset_sigma=getattr(args, "offset_sigma", None))
    val_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='val',
                            with_depth=bool(getattr(args, "with_depth", False)),
                            version=args.version)


    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=bool(getattr(args, "pin_memory", False)),
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True,
                                   collate_fn=OCIDVLGDataset.collate_fn)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=bool(getattr(args, "pin_memory", False)),
                                 sampler=val_sampler,
                                 drop_last=False,
                                 collate_fn=OCIDVLGDataset.collate_fn)

    best_IoU = -1.0
    best_j_index = -1.0
    last_eval_epoch = 0
    last_iou = None
    last_prec_dict = {}
    last_j_index = []
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            best_j_index = checkpoint["best_j_index"]
            last_eval_epoch = int(
                checkpoint.get("last_eval_epoch", checkpoint["epoch"])
            )
            last_iou = checkpoint.get(
                "last_iou", checkpoint.get("cur_iou")
            )
            last_prec_dict = checkpoint.get(
                "last_prec", checkpoint.get("prec", {})
            )
            last_j_index = checkpoint.get(
                "last_j_index", checkpoint.get("j_index", [])
            )
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group["foreach"] = optimizer_foreach
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(args.device)
            scheduler = rebuild_multistep_scheduler(
                optimizer=optimizer,
                base_lrs=configured_base_lrs,
                milestones=args.milestones,
                gamma=args.lr_decay,
                completed_epochs=args.start_epoch,
            )
            logger.info(
                "=> rebuilt LR schedule from current config: "
                "milestones={}, gamma={}, completed_epochs={}, lr={}",
                args.milestones,
                args.lr_decay,
                args.start_epoch,
                [group["lr"] for group in optimizer.param_groups],
            )
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))

            del checkpoint
            empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    val_freq = int(getattr(args, "val_freq", 1))
    val_start_epoch = int(getattr(args, "val_start_epoch", 1))
    save_epochs = tuple(
        sorted({int(epoch) for epoch in getattr(args, "save_epochs", [])})
    )
    evaluate_enabled = bool(getattr(args, "evaluate", True))
    if evaluate_enabled and val_freq <= 0:
        raise ValueError(f"val_freq must be positive, got {val_freq}")
    if val_start_epoch <= 0:
        raise ValueError(
            f"val_start_epoch must be positive, got {val_start_epoch}"
        )
    invalid_save_epochs = [
        epoch for epoch in save_epochs
        if epoch <= 0 or epoch > args.epochs
    ]
    if invalid_save_epochs:
        raise ValueError(
            "save_epochs must be within the configured training range "
            f"1..{args.epochs}, got {invalid_save_epochs}"
        )
    if args.rank == 0:
        logger.info(
            "Validation schedule: enabled={}, starts at epoch {}, every {} "
            "epoch(s); recovery checkpoints={}; checkpoint policy=scheduled "
            "recovery + one metric-labelled best",
            evaluate_enabled,
            val_start_epoch,
            val_freq,
            list(save_epochs),
        )

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch_log,  args)
        do_eval = evaluate_enabled and epoch_log >= val_start_epoch and (
            (epoch_log - val_start_epoch) % val_freq == 0
        )
        iou, prec_dict, j_index = None, {}, []
        segmentation_only = bool(
            getattr(model.module, "segmentation_only", False)
        )
        if do_eval:
            # Official MapleGrasp Stage 1 validates segmentation only; Stage 2
            # validates segmentation plus grasp maps through the CROG scorer.
            if segmentation_only:
                iou, prec_dict, j_index = validate_without_grasp(
                    val_loader, model, epoch_log, args
                )
            elif args.use_grasp_masks:
                iou, prec_dict, j_index = validate_with_grasp(
                    val_loader, model, epoch_log, args
                )
            else:
                iou, prec_dict, j_index = validate_without_grasp(
                    val_loader, model, epoch_log, args
                )
            last_eval_epoch = epoch_log
            last_iou = iou
            last_prec_dict = prec_dict
            last_j_index = j_index
        elif args.rank == 0:
            logger.info(
                "Skipping validation at epoch {} (starts at {}, val_freq={})",
                epoch_log,
                val_start_epoch,
                val_freq,
            )

        # Keep the checkpoint scheduler state aligned with the next epoch.
        # This is the same uninterrupted training schedule as upstream.
        scheduler.step(epoch_log)

        # Save only explicit recovery epochs and newly improved best models.
        if dist.get_rank() == 0:
            improved_iou = bool(do_eval and iou > best_IoU)
            improved_j = bool(
                do_eval and j_index and j_index[0] > best_j_index
            )
            if improved_iou:
                best_IoU = iou
            if improved_j:
                best_j_index = j_index[0]

            save_recovery = epoch_log in save_epochs
            save_best_iou = bool(
                do_eval and segmentation_only and improved_iou
            )
            save_best_j = bool(
                do_eval and not segmentation_only and improved_j
            )
            if save_recovery or save_best_iou or save_best_j:
                checkpoint = {
                    'epoch': epoch_log,
                    'base_exp_name': args.base_exp_name,
                    'run_timestamp': args.run_timestamp,
                    'output_dir': args.output_dir,
                    'evaluated': do_eval,
                    'cur_iou': iou,
                    'best_iou': best_IoU,
                    'best_j_index': best_j_index,
                    'prec': prec_dict,
                    'j_index': j_index,
                    'last_eval_epoch': last_eval_epoch,
                    'last_iou': last_iou,
                    'last_prec': last_prec_dict,
                    'last_j_index': last_j_index,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                temporary_checkpoint = os.path.join(
                    args.output_dir, ".checkpoint.pth.tmp"
                )
                torch.save(checkpoint, temporary_checkpoint)

                if save_recovery:
                    recovery_name = os.path.join(
                        args.output_dir,
                        f"epoch_{epoch_log:03d}_model.pth",
                    )
                    _replace_with_link_or_copy(
                        temporary_checkpoint, recovery_name
                    )
                    logger.info(
                        "Saved scheduled recovery checkpoint: {}",
                        recovery_name,
                    )

                if save_best_iou:
                    best_name = _replace_metric_alias(
                        temporary_checkpoint,
                        args.output_dir,
                        "best",
                        epoch_log,
                        f"IoU_{100.0 * float(iou):.2f}",
                    )
                    logger.info("Replaced best checkpoint: {}", best_name)

                if save_best_j:
                    j1 = float(j_index[0])
                    j5 = float(j_index[1]) if len(j_index) > 1 else 0.0
                    best_name = _replace_metric_alias(
                        temporary_checkpoint,
                        args.output_dir,
                        "best",
                        epoch_log,
                        f"J1_{100.0 * j1:.2f}_J5_{100.0 * j5:.2f}",
                    )
                    logger.info("Replaced best checkpoint: {}", best_name)

                os.remove(temporary_checkpoint)
        empty_cache()

    time.sleep(2)
    # if dist.get_rank() == 0:
    #     wandb.finish()

    logger.info("* Best IoU={}  Best J1={} *".format(
        best_IoU, best_j_index
    ))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
    sys.exit(0)
