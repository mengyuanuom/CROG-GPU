"""Check the CUDA runtime without constructing the full CROG model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from utils.cuda import device_count, set_device


def main() -> int:
    device = set_device(0)
    print(f"torch={torch.__version__}")
    print(f"torch CUDA runtime={torch.version.cuda}")
    print(f"cuDNN={torch.backends.cudnn.version()}")
    print(f"visible CUDA devices={device_count()}")
    print(f"device={device}")
    print(f"device name={torch.cuda.get_device_name(device)}")
    print(f"NCCL available={torch.distributed.is_nccl_available()}")
    left = torch.randn(64, 64, device=device)
    right = torch.randn(64, 64, device=device)
    result = left @ right
    torch.cuda.synchronize(device)
    print(f"CUDA matmul OK: shape={tuple(result.shape)} dtype={result.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
