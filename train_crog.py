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


def _replace_epoch_alias(source, output_dir, prefix, epoch):
    """Keep one epoch-labelled alias for a moving checkpoint series."""
    output_dir = Path(output_dir)
    target = output_dir / f"{prefix}_epoch_{int(epoch):03d}.pth"
    _replace_with_link_or_copy(source, target)
    for previous in output_dir.glob(f"{prefix}_epoch_*.pth"):
        if previous != target:
            previous.unlink()
    return target


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
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # local rank & global rank
    args.npu = local_rank
    args.device = set_device(local_rank)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.rank,
                 filename="train.log",
                 mode="a")

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
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    amp_enabled = bool(getattr(args, "amp", True))
    scaler = build_grad_scaler(enabled=amp_enabled)
    if args.rank == 0:
        logger.info(
            "Precision path: amp={}, scaler={}",
            amp_enabled,
            type(scaler).__name__,
        )

    # # resume
    # best_IoU = 0.0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(
    #             args.resume, map_location=torch.device('cpu'))
    #         args.start_epoch = checkpoint['epoch']
    #         best_IoU = checkpoint["best_iou"]
    #         state_dict = checkpoint['state_dict']
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             name = k[7:] # remove `module.`
    #             new_state_dict[name] = v
    #         # load params
    #         model.load_state_dict(new_state_dict)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #             args.resume, checkpoint['epoch']))
    #     else:
    #         raise ValueError(
    #             "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
    #             .format(args.resume))



    model = model.to(args.device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.npu],
        output_device=args.npu,
        find_unused_parameters=True,
    )

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

    best_IoU = 0.0
    best_j_index = 0.0
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
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(args.device)
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))

            del checkpoint
            empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    val_freq = int(getattr(args, "val_freq", 1))
    save_freq = int(getattr(args, "save_freq", 1))
    evaluate_enabled = bool(getattr(args, "evaluate", True))
    if evaluate_enabled and val_freq <= 0:
        raise ValueError(f"val_freq must be positive, got {val_freq}")
    if save_freq < 0:
        raise ValueError(f"save_freq must be non-negative, got {save_freq}")
    if args.rank == 0:
        logger.info(
            "Validation schedule: enabled={}, every {} epoch(s), "
            "always on final epoch; checkpoint snapshots every {} epoch(s)",
            evaluate_enabled,
            val_freq,
            save_freq if save_freq else "disabled",
        )

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch_log,  args)
        do_eval = evaluate_enabled and (
            epoch_log % val_freq == 0 or epoch_log == args.epochs
        )
        iou, prec_dict, j_index = None, {}, []
        if do_eval:
            if args.use_grasp_masks:
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
                "Skipping validation at epoch {} (val_freq={})",
                epoch_log,
                val_freq,
            )

        # Keep the checkpoint scheduler state aligned with the next epoch.
        # This is the same uninterrupted training schedule as upstream.
        scheduler.step(epoch_log)

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            improved_iou = bool(do_eval and iou >= best_IoU)
            improved_j = bool(
                do_eval and j_index and j_index[0] >= best_j_index
            )
            if improved_iou:
                best_IoU = iou
            if improved_j:
                best_j_index = j_index[0]
            checkpoint = {
                'epoch': epoch_log,
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
            temporary_lastname = os.path.join(
                args.output_dir, ".last_model.pth.tmp"
            )
            torch.save(checkpoint, temporary_lastname)
            os.replace(temporary_lastname, lastname)
            _replace_epoch_alias(
                lastname, args.output_dir, "last", epoch_log
            )

            if save_freq and (
                epoch_log % save_freq == 0 or epoch_log == args.epochs
            ):
                epoch_name = os.path.join(
                    args.output_dir,
                    f"epoch_{epoch_log:03d}_model.pth",
                )
                _replace_with_link_or_copy(lastname, epoch_name)

            if improved_iou:
                bestname = _replace_epoch_alias(
                    lastname, args.output_dir, "best_iou", epoch_log
                )
                _replace_with_link_or_copy(
                    bestname,
                    os.path.join(args.output_dir, "best_iou_model.pth"),
                )

            if improved_j:
                bestname = _replace_epoch_alias(
                    lastname, args.output_dir, "best_jindex", epoch_log
                )
                _replace_with_link_or_copy(
                    bestname,
                    os.path.join(args.output_dir, "best_jindex_model.pth"),
                )

        empty_cache()

    time.sleep(2)
    # if dist.get_rank() == 0:
    #     wandb.finish()

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
    sys.exit(0)
