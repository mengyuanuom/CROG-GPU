# CROG-GPU migration contract

`CROG-GPU` is derived from the latest `CROG-NPU` main branch. The conversion
keeps algorithmic behavior and changes only accelerator integration and the
GPU-oriented defaults described below.

## Preserved from CROG-NPU

- CROG, CROG-OFF, DROG, DROG-OFF, ETRG, GraspMamba, LGD, MapleGrasp,
  GGCNN-CLIP and GR-ConvNet-CLIP model paths.
- OCID-VLG, Grasp-Tools and VCoT/Grasp-Anything dataset support.
- Global batch-size semantics, gradient accumulation and exact distributed
  evaluation sharding.
- CROG legacy and VCoT official evaluation protocols.
- Timestamped runs, scheduled recovery checkpoints and metric-labelled best
  checkpoints.
- Verified pretrained-weight download and resume behavior.

## CUDA substitutions

| CROG-NPU | CROG-GPU |
| --- | --- |
| `torch_npu` runtime | `torch.cuda` runtime |
| HCCL | NCCL |
| `ASCEND_RT_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` |
| NPU AMP/scaler | `torch.cuda.amp` |
| NPU environment check | `tools/check_cuda_env.py` |
| eight-NPU launcher | visible-GPU-aware `tools/train_gpu.sh` |
| NPU-safe FP32 defaults | CUDA AMP, pinned memory and Adam foreach defaults |

The DINOv2 path uses xFormers CUDA kernels when xFormers is installed and
falls back to standard PyTorch operations otherwise. Set
`XFORMERS_DISABLED=1` to force the fallback.

## Validation

The conversion is covered by source compilation and 64 unit/contract tests.
Run them with:

```bash
python -m compileall -q train_crog.py test_crog.py engine model utils tools
python -m unittest discover -s tests -p 'test_*.py'
```

A real CUDA smoke run still requires a Linux host with NVIDIA drivers, a
CUDA-enabled PyTorch installation and the pretrained weights/datasets used by
the selected YAML profile:

```bash
pip install -r requirements-gpu.txt
python tools/check_cuda_env.py
CUDA_VISIBLE_DEVICES=0 bash tools/train_gpu.sh config/OCID-VLG/crog_multiple_r50.yaml
```
