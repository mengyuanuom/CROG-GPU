import os
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU, get_seg_image)
from utils.npu import autocast
from utils.grasp_eval import (detect_grasps, calculate_iou, calculate_max_iou, calculate_jacquard_index, visualization)
from utils.offset_eval import refine_with_offset, resample_grasp_geometry

def _apply_model_offset(
    grasps,
    offset_map,
    inverse_matrix,
    sine_map,
    cosine_map,
    width_map,
    input_hw,
    args,
):
    """Apply DROG-OFF post-processing before the unchanged CROG scorer."""
    if offset_map is None or not grasps:
        return grasps
    radius = float(getattr(args, "offset_r", min(input_hw) / 20.0))
    refined = refine_with_offset(
        grasps, offset_map, inverse_matrix, max(1.0, radius)
    )
    if bool(getattr(args, "offset_resample_geometry", False)):
        refined = resample_grasp_geometry(
            refined,
            sine_map,
            cosine_map,
            width_map,
            width_factor=100.0,
        )
    return refined

def train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    qua_loss_metter = AverageMeter('Loss_qua', ':2.4f')
    sin_loss_metter = AverageMeter('Loss_sin', ':2.4f')
    cos_loss_metter = AverageMeter('Loss_cos', ':2.4f')
    wid_loss_metter = AverageMeter('Loss_wid', ':2.4f')
    unwrapped_model = getattr(model, "module", model)
    off_loss_metter = (
        AverageMeter('Loss_off', ':2.4f')
        if bool(getattr(unwrapped_model, "supports_offset", False))
        else None
    )
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    component_loss_meters = [
        qua_loss_metter,
        sin_loss_metter,
        cos_loss_metter,
        wid_loss_metter,
    ]
    if off_loss_metter is not None:
        component_loss_meters.append(off_loss_metter)
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr, loss_meter,
            *component_loss_meters,
            iou_meter, pr_meter
        ],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, data in enumerate(train_loader):
        # image, target, text = data
        # ins_mask, grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_width_mask = target

        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        grasp_off_mask = data["grasp_masks"].get("off")
        grasp_off_weight = data["grasp_masks"].get("off_w")


        data_time.update(time.time() - end)
        # data
        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)
        if grasp_off_mask is not None:
            grasp_off_mask = grasp_off_mask.to(args.device, non_blocking=True)
            grasp_off_weight = grasp_off_weight.to(args.device, non_blocking=True)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        with autocast(enabled=bool(getattr(args, "amp", True))):
            model_inputs = (
                image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
                grasp_cos_mask, grasp_wid_mask,
            )
            if grasp_off_mask is not None:
                model_inputs = (*model_inputs, grasp_off_mask, grasp_off_weight)
            pred, target, loss, loss_dict = model(*model_inputs)

        ins_mask_pred = pred[0]
        ins_mask_target = target[0]

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(ins_mask_pred, ins_mask_target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        qua_loss_metter.update(loss_dict["m_qua"], image.size(0))
        sin_loss_metter.update(loss_dict["m_sin"], image.size(0))
        cos_loss_metter.update(loss_dict["m_cos"], image.size(0))
        wid_loss_metter.update(loss_dict["m_wid"], image.size(0))
        if off_loss_metter is not None:
            off_loss_metter.update(loss_dict["m_off"], image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            # if dist.get_rank() in [-1, 0]:
            #     wandb.log(
            #         {
            #             "time/batch": batch_time.val,
            #             "time/data": data_time.val,
            #             "training/lr": lr.val,
            #             "training/loss": loss_meter.val,
            #             "training/loss_qua": qua_loss_metter.val,
            #             "training/loss_sin": sin_loss_metter.val,
            #             "training/loss_cos": cos_loss_metter.val,
            #             "training/loss_wid": wid_loss_metter.val,
            #             "training/iou": iou_meter.val,
            #             "training/prec@50": pr_meter.val,
            #         },
            #         step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate_with_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader, disable=dist.get_rank() != 0)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]

        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)

        # inference & get predictions from model
        pred, _ = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]
        grasp_off_mask_preds = pred[5] if len(pred) > 5 else None

        # Evaluation uses original input-resolution dataloader targets.
        # Model-returned targets may be resized for training loss computation.
        ins_mask_targets = ins_mask
        grasp_qua_mask_targets = grasp_qua_mask
        grasp_sin_mask_targets = grasp_sin_mask
        grasp_cos_mask_targets = grasp_cos_mask
        grasp_wid_mask_targets = grasp_wid_mask

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)
        if (
            grasp_off_mask_preds is not None
            and grasp_off_mask_preds.shape[-2:] != image.shape[-2:]
        ):
            grasp_off_mask_preds = F.interpolate(
                grasp_off_mask_preds,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size

            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            grasp_off_mask_pred = (
                grasp_off_mask_preds[idx : idx + 1].cpu().numpy()
                if grasp_off_mask_preds is not None
                else None
            )

            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, _ = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)
                grasp_preds = _apply_model_offset(
                    grasp_preds,
                    grasp_off_mask_pred,
                    inv_mat,
                    grasp_sin_mask_pred,
                    grasp_cos_mask_pred,
                    grasp_wid_mask_pred,
                    image.shape[-2:],
                    args,
                )

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)

                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1

    correct = torch.tensor(
        num_correct_grasps, dtype=torch.float32, device=image.device
    )
    total = torch.tensor(
        num_total_grasps, dtype=torch.float32, device=image.device
    )
    dist.all_reduce(correct)
    dist.all_reduce(total)
    J_index = (correct / total.clamp_min(1)).cpu().tolist()

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index


@torch.no_grad()
def validate_without_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader, disable=dist.get_rank() != 0)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]

        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)

        # inference & get predictions from model
        pred, _ = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)
        ins_mask_targets = ins_mask

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(pred)
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size

            ins_mask_pred = ins_mask_preds[idx].squeeze().cpu().numpy()
            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

    J_index = [0, 0]

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index



@torch.no_grad()
def inference_with_grasp(test_loader, model, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
    iou_sum = 0.0
    sample_count = 0
    precision_counts = [0, 0, 0, 0, 0]
    model.eval()
    time.sleep(2)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    tbar = tqdm(
        test_loader,
        desc='Inference:',
        ncols=100,
        disable=rank != 0,
    )
    for cnt, data in enumerate(tbar):

        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        sentences = data["sentence"]
        img_paths = data["img_path"]

        image = image.to(args.device, non_blocking=True)
        text = text.to(args.device, non_blocking=True)
        ins_mask = ins_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.to(args.device, non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.to(args.device, non_blocking=True).unsqueeze(1)

        # inference & get predictions from model
        pred, _ = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]
        grasp_off_mask_preds = pred[5] if len(pred) > 5 else None

        # Evaluation uses original input-resolution dataloader targets.
        # Model-returned targets may be resized for training loss computation.
        ins_mask_targets = ins_mask
        grasp_qua_mask_targets = grasp_qua_mask
        grasp_sin_mask_targets = grasp_sin_mask
        grasp_cos_mask_targets = grasp_cos_mask
        grasp_wid_mask_targets = grasp_wid_mask

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)
        if (
            grasp_off_mask_preds is not None
            and grasp_off_mask_preds.shape[-2:] != image.shape[-2:]
        ):
            grasp_off_mask_preds = F.interpolate(
                grasp_off_mask_preds,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)


        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            sent = sentences[idx]
            img_path = img_paths[idx]

            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            grasp_off_mask_pred = (
                grasp_off_mask_preds[idx : idx + 1].cpu().numpy()
                if grasp_off_mask_preds is not None
                else None
            )

            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_sum += float(iou)
            sample_count += 1
            for threshold_idx, threshold in enumerate(thresholds):
                precision_counts[threshold_idx] += int(iou > threshold)

            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, grasp_ang_mask_pred = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)
                grasp_preds = _apply_model_offset(
                    grasp_preds,
                    grasp_off_mask_pred,
                    inv_mat,
                    grasp_sin_mask_pred,
                    grasp_cos_mask_pred,
                    grasp_wid_mask_pred,
                    image.shape[-2:],
                    args,
                )

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)

                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1

                # Visualization
                if args.visualize:
                    img_bgr = cv2.imread(img_path)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    visualization(img, ins_mask_pred, (grasp_qua_mask_pred, grasp_ang_mask_pred, grasp_wid_mask_pred), grasp_preds, sent, save_path=os.path.join("./results", args.exp_name, f"results_rank{rank}_{cnt}_{num_g}_grasps.png"))

    stats = torch.tensor(
        [
            iou_sum,
            sample_count,
            *precision_counts,
            num_correct_grasps[0],
            num_total_grasps[0],
            num_correct_grasps[1],
            num_total_grasps[1],
        ],
        dtype=torch.float32,
        device=args.device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total = stats[1].clamp_min(1.0)
    iou = stats[0] / total
    prec = {
        "Pr@{}".format(int(threshold * 100)): (
            stats[2 + index] / total
        ).item()
        for index, threshold in enumerate(thresholds)
    }
    J_index = [
        (stats[7] / stats[8].clamp_min(1.0)).item(),
        (stats[9] / stats[10].clamp_min(1.0)).item(),
    ]
    if rank == 0:
        logger.info('IoU={:.2f}'.format(100.*iou.item()))
        for key, value in prec.items():
            logger.info('{}: {:.2f}.'.format(key, 100.*value))
        logger.info(
            "J@1: {:.2f}, J@5: {:.2f}".format(
                100. * J_index[0], 100. * J_index[1]
            )
        )

    return iou.item(), prec, J_index
