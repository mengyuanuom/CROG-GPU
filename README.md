# Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter (CoRL2023)

Created by Georgios Tziafas, Yucheng XU, Arushi Goel, Mohammadreza Kasaei, Zhibin Li, Hamidreza Kasaei

This is an official PyTorch implementation of the baseline end-to-end model [CROG](https://arxiv.org/abs/2311.05779) of our work. The implementation of our CROG model is based on the [CRIS](https://github.com/DerrickWang005/CRIS.pytorch) model, thanks for their amazing work! :beers:

Robots operating in human-centric environments require the integration of visual grounding and grasping capabilities to effectively manipulate objects based on user instructions. This work focuses on the task of referring grasp synthesis, which predicts a grasp pose for an object referred through natural language in cluttered scenes. Existing approaches often employ multi-stage pipelines that first segment the referred object and then propose a suitable grasp, and are evaluated in private datasets or simulators that do not capture the complexity of natural indoor scenes. To address these limitations, we develop a challenging benchmark based on cluttered indoor scenes from OCID dataset, for which we generate referring expressions and connect them with 4-DoF grasp poses. Further, we propose a novel end-to-end model (CROG) that leverages the visual grounding capabilities of CLIP to learn grasp synthesis directly from image-text pairs. Our results show that vanilla integration of CLIP with pretrained models transfers poorly in our challenging benchmark, while CROG achieves significant improvements both in terms of grounding and grasping. Extensive robot experiments in both simulation and hardware demonstrate the effectiveness of our approach in challenging interactive object grasping scenarios that include clutter.


**Check our demo video [here](https://www.youtube.com/watch?v=D3auLBUX-EM&t=5s)**

## Example
<p align="center">
  <img src="media/example.png" width="600">
</p>


## News
- :sunny: [Aug 30, 2023] Our paper was accepted by CoRL-2023.


## Preparation

1. Environment
   - use the environment.yml file to create the conda env.
2. Datasets
   - The detailed instruction is in [OCID-VLG](https://github.com/gtziafas/OCID-VLG) repo.

## Quick Start

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported. Besides, the evaluation only supports single-gpu mode. In our case, we train the CROG on 2 RTX-4090 GPUs. The training procedure takes around 3.5 hours. To do training of CROG with 2 GPUs, run:

```
python -u train_crog.py --config config/OCID-VLG/CROG_multiple_r50.yaml
```

To do training of SSG with 2 GPUs, run:
```
python -u train_ssg.py --config config/OCID-Grasp/ssg_r50.yaml
```

**Please remember to modify the path to the dataset in config files.**

## Ascend NPU port

The CROG and DROG configurations keep the official global batch size (24),
Adam learning rate (`1e-4`), 50-epoch schedule, and epoch-35 learning-rate
decay. DROG-OFF uses a global batch size of 128 and learning rate `4e-4`.
Only the accelerator/runtime path is changed:

- explicit `torch_npu` device calls instead of CUDA calls;
- HCCL and `torchrun` instead of NCCL and the in-process GPU launcher;
- full FP32 on Ascend because AMP caused gradient overflow;
- per-rank BatchNorm instead of SyncBatchNorm. The latter is deliberately
  disabled because torch_npu SyncBatchNorm can produce device-side
  AIVector/MTE faults during multi-NPU training;
- CPU checkpoint loading followed by explicit optimizer-state migration.

Install the PyTorch/torch_npu pair matching the server's CANN release, then:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r requirements-npu.txt
python tools/check_npu_env.py
```

Place OCID-VLG at `datasets/OCID-VLG`. CROG is RGB-only, so the training
dataset does not need a `depth/` directory and depth images are never loaded.
The training launcher automatically
downloads the official OpenAI CLIP RN50 checkpoint to `pretrain/RN50.pt` when
it is absent and verifies its SHA-256 before training. Run the original CROG
experiment on eight NPUs with:

```bash
bash tools/train_crog_8npu.sh
```

The YAML fixes `TRAIN.amp: False`, and the launcher does not override it:

```bash
bash tools/train_crog_8npu.sh
```

The FP32 path bypasses both autocast and `torch_npu.npu.amp.GradScaler`; it
uses ordinary `loss.backward()` and `optimizer.step()` so a disabled scaler
cannot still enter an Ascend overflow-status check.

Evaluate a CROG checkpoint on one NPU with:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 test_crog.py \
  --config config/OCID-VLG/crog_multiple_r50.yaml \
  --opts DATA.root_path datasets/OCID-VLG \
         TRAIN.resume exp/OCID-VLG_multiple_npu/CROG_official_multiple_R50_8npu/best_jindex_model.pth \
         TEST.test_split test
```

For eight-NPU evaluation, the dataset is sharded without padding and the
metrics are summed with HCCL:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 test_crog.py \
  --config config/OCID-VLG/crog_multiple_r50.yaml \
  --opts DATA.root_path datasets/OCID-VLG \
         TRAIN.resume exp/OCID-VLG_multiple_npu/CROG_official_multiple_R50_8npu/best_jindex_model.pth \
         TEST.test_split test
```



## DROG and DROG-OFF with the CROG scoring protocol

`config/OCID-VLG/drog.yaml` and `config/OCID-VLG/drogoff.yaml` select the
DINOv2/CLIP-B16 models. DROG-OFF keeps its offset post-processing, while all
resulting grasp rectangles are judged by CROG's scoring functions. Put these
pretrained files under `pretrain/`:

- `ViT-B-16.pt`
- `dinov2_vitb14_reg4_pretrain.pth`

Train DROG on eight NPUs:

```bash
bash tools/train_drog_8npu.sh
```

Train DROG-OFF with the same launcher:

```bash
CONFIG=config/OCID-VLG/drogoff.yaml bash tools/train_drog_8npu.sh
```

Evaluate a DROG checkpoint on one NPU:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 test_crog.py \
  --config config/OCID-VLG/drog.yaml \
  --opts TRAIN.resume exp/OCID-VLG/drog_ocid_vlg_8npu/best_jindex_model.pth
```

Use `config/OCID-VLG/drogoff.yaml` and the matching checkpoint for DROG-OFF.
Eight-NPU evaluation uses the same `torchrun --nproc_per_node=8 test_crog.py`
form documented above for CROG.

This comparison intentionally preserves every historical CROG evaluation
operation: bicubic resizing with `align_corners=True`, OpenCV cubic inverse
warping, the 0.35 segmentation threshold, quality peaks at 0.4 with distance
2, top-1/top-5 detection, the fixed 480x640 raster canvas, predicted grasp
height 20, ground-truth height overwritten to 20, ground-truth width clipped
to 100, the original periodic-angle test, and strict `IoU > 0.25`. DROG-OFF's
offset head is supervised during training and refines the predicted center (plus
angle/width resampling) before the resulting rectangle is passed to the unchanged
CROG Jacquard scorer.
The weight can also be downloaded separately:

```bash
python tools/download_clip_rn50.py
```

Official direct URL:
<https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt>

Custom locations can be supplied without editing files:

```bash
DATA_ROOT=/data/OCID-VLG CLIP_WEIGHT=/data/RN50.pt \
  bash tools/train_crog_8npu.sh
```


## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{tziafas2023language,
  title={Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter},
  author={Tziafas, Georgios and Yucheng, XU and Goel, Arushi and Kasaei, Mohammadreza and Li, Zhibin and Kasaei, Hamidreza},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}

@inproceedings{10161149,
  author={Xu, Yucheng and Kasaei, Mohammadreza and Kasaei, Hamidreza and Li, Zhibin},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  title={Instance-wise Grasp Synthesis for Robotic Grasping},
  year={2023},
  volume={},
  number={},
  pages={1744-1750},
  keywords={Automation;Object detection;Grasping;Benchmark testing;Feature extraction;Cleaning;Task analysis},
  doi={10.1109/ICRA48891.2023.10161149}}
```
