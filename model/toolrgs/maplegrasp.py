"""MapleGrasp mask-guided dense grasp model for ToolRGSNPU.

This implementation adapts the official CROG-based MapleGrasp head to the
shared ToolRGS five-map contract. It keeps the detached predicted-mask gating
between segmentation and grasp prediction, avoids fixed spatial sizes, and
does not consume ground-truth masks during evaluation. Device placement is
owned by the NPU runner; this module intentionally contains no CUDA calls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crog import CROG
from .crog_layers import conv_layer


class MapleGraspProjector(nn.Module):
    """Dynamic text projector with mask-guided grasp feature pooling."""

    def __init__(
        self,
        word_dim=1024,
        in_dim=256,
        kernel_size=3,
        mask_threshold=0.35,
        hard_mask=True,
        use_gt_masks=False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.kernel_size = int(kernel_size)
        self.mask_threshold = float(mask_threshold)
        self.hard_mask = bool(hard_mask)
        self.use_gt_masks = bool(use_gt_masks)

        self.visual = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
        )
        self.mask_features = nn.Conv2d(in_dim, in_dim, 1)
        self.grasp_features = nn.Conv2d(in_dim, in_dim * 4, 1)
        output_dim = in_dim * kernel_size * kernel_size + 1
        self.text_kernel = nn.Linear(word_dim, output_dim)

    def _kernel(self, text_state):
        batch_size = text_state.shape[0]
        parameters = self.text_kernel(text_state)
        weight, bias = parameters[:, :-1], parameters[:, -1]
        weight = weight.reshape(
            batch_size,
            self.in_dim,
            self.kernel_size,
            self.kernel_size,
        )
        return weight, bias

    def _dynamic_conv(self, features, weight, bias):
        batch_size, channels, height, width = features.shape
        features = features.reshape(1, batch_size * channels, height, width)
        output = F.conv2d(
            features,
            weight,
            bias=bias,
            padding=self.kernel_size // 2,
            groups=batch_size,
        )
        return output.transpose(0, 1)

    def _gate(self, segmentation, target_mask, output_size):
        gate = torch.sigmoid(segmentation.detach())
        if self.hard_mask:
            gate = (gate > self.mask_threshold).to(gate.dtype)
        if self.training and self.use_gt_masks and target_mask is not None:
            gate = F.interpolate(
                target_mask.detach().to(gate.dtype),
                size=output_size,
                mode="nearest",
            )
        elif gate.shape[-2:] != output_size:
            gate = F.interpolate(
                gate,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return gate

    def forward(self, features, text_state, target_mask=None):
        visual = self.visual(features)
        weight, bias = self._kernel(text_state)
        segmentation = self._dynamic_conv(self.mask_features(visual), weight, bias)

        grasp_features = self.grasp_features(visual)
        gate = self._gate(segmentation, target_mask, grasp_features.shape[-2:])
        grasp_features = grasp_features * gate
        grasp_features = torch.chunk(grasp_features, 4, dim=1)
        grasp_outputs = tuple(
            self._dynamic_conv(item, weight, bias) for item in grasp_features
        )
        return (segmentation, *grasp_outputs)


class MapleGrasp(CROG):
    """CROG backbone plus MapleGrasp mask-guided grasp prediction."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.use_grasp_masks:
            raise ValueError("MapleGrasp requires use_grasp_masks=True")
        self.maple_stage = str(getattr(cfg, "maple_stage", "joint")).lower()
        if self.maple_stage not in {"segmentation", "grasp", "joint"}:
            raise ValueError("maple_stage must be one of: segmentation, grasp, joint")
        self.grasp_loss_weight = float(
            getattr(cfg, "maple_grasp_loss_weight", 1.0)
        )
        self.proj = MapleGraspProjector(
            word_dim=cfg.word_dim,
            in_dim=cfg.vis_dim // 2,
            kernel_size=3,
            mask_threshold=float(getattr(cfg, "maple_mask_threshold", 0.35)),
            hard_mask=bool(getattr(cfg, "maple_hard_mask", True)),
            use_gt_masks=bool(getattr(cfg, "maple_use_gt_masks", False)),
        )

    @staticmethod
    def _resize_target(target, size):
        if target is None:
            raise ValueError("MapleGrasp training requires all five dense targets")
        return F.interpolate(target, size=size, mode="nearest").detach()

    def forward(
        self,
        img,
        word,
        mask=None,
        grasp_qua_mask=None,
        grasp_sin_mask=None,
        grasp_cos_mask=None,
        grasp_wid_mask=None,
        grasp_off_mask=None,
        grasp_off_weight=None,
    ):
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        visual, word_features, text_state = self._encode(img, word)
        features = self.neck(visual, text_state)
        batch_size, channels, height, width = features.shape
        if self.use_contrastive:
            features = self.decoder(features, word_features, pad_mask)
            features = features.reshape(batch_size, channels, height, width)

        outputs = self.proj(features, text_state, target_mask=mask)
        if mask is None:
            return outputs

        output_size = outputs[0].shape[-2:]
        targets = tuple(
            self._resize_target(target, output_size)
            for target in (
                mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
            )
        )
        if not self.training:
            return tuple(item.detach() for item in outputs), targets

        segmentation, quality, sine, cosine, grasp_width = outputs
        seg_weight = targets[0] * 0.5 + 1.0
        segmentation_loss = F.binary_cross_entropy_with_logits(
            segmentation, targets[0], weight=seg_weight
        )
        losses = {
            "m_ins": segmentation_loss.detach(),
            "m_qua": segmentation_loss.new_zeros(()),
            "m_sin": segmentation_loss.new_zeros(()),
            "m_cos": segmentation_loss.new_zeros(()),
            "m_wid": segmentation_loss.new_zeros(()),
        }
        total_loss = segmentation_loss

        if self.maple_stage != "segmentation":
            grasp_losses = (
                F.smooth_l1_loss(quality, targets[1]),
                F.smooth_l1_loss(sine, targets[2]),
                F.smooth_l1_loss(cosine, targets[3]),
                F.smooth_l1_loss(grasp_width, targets[4]),
            )
            total_loss = total_loss + self.grasp_loss_weight * sum(grasp_losses)
            for name, value in zip(
                ("m_qua", "m_sin", "m_cos", "m_wid"), grasp_losses
            ):
                losses[name] = value.detach()

        return (
            tuple(item.detach() for item in outputs),
            targets,
            total_loss,
            losses,
        )

    def _encode(self, image, words):
        visual = self.backbone.encode_image(image)
        word_features, text_state = self.backbone.encode_text(words)
        return visual, word_features, text_state
