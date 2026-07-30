import argparse
import os
import warnings

import cv2
import torch
import torch.distributed as dist
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.crog_engine import inference_with_grasp
from model import build_model
from utils.dataset import OCIDVLGDataset
from utils.misc import setup_logger
from utils.npu import set_device

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


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
            "test_crog.py preserves the CROG evaluation protocol; "
            "set TEST.evaluation_protocol=crog_legacy."
        )
    args.evaluation_protocol = "crog_legacy"
    args.npu = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1
    args.device = set_device(args.npu)
    if args.distributed:
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=args.rank,
            world_size=args.world_size,
        )
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.rank,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build dataset & dataloader
    test_split = getattr(args, "test_split", "test")
    if test_split == "val-test":
        logger.warning(
            "test_split='val-test' is not an OCID-VLG split; using 'test'."
        )
        test_split = "test"
    if test_split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unsupported OCID-VLG test split: {test_split!r}. "
            "Choose train, val, or test."
        )
    full_test_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split=test_split,
                            with_depth=bool(getattr(args, "with_depth", False)),
                            version=args.version)
    if args.distributed:
        indices = range(args.rank, len(full_test_data), args.world_size)
        test_data = torch.utils.data.Subset(full_test_data, indices)
    else:
        test_data = full_test_data
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=bool(
                                                  getattr(args, "pin_memory", False)
                                              ),
                                              collate_fn=OCIDVLGDataset.collate_fn)

    # build model
    model, _ = build_model(args)
    model = model.to(args.device)
    logger.info(model)

    save_path = os.path.join("./results", args.exp_name)
    os.makedirs(save_path, exist_ok=True)

    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.resume))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.resume))

    # inference
    try:
        inference_with_grasp(test_loader, model, args)
    finally:
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
