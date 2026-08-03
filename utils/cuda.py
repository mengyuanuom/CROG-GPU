"""Small, explicit CUDA runtime used by the CROG training code."""

from contextlib import nullcontext
import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build and "
            "check the NVIDIA driver with nvidia-smi."
        )


def device_count() -> int:
    require_cuda()
    return int(torch.cuda.device_count())


def set_device(index: int) -> torch.device:
    require_cuda()
    index = int(index)
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def autocast(enabled: bool = True):
    if not enabled:
        return nullcontext()
    return torch.cuda.amp.autocast(enabled=True)


class NoOpGradScaler:
    """FP32 optimizer path with the same interface as GradScaler."""

    enabled = False

    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        return None

    @staticmethod
    def step(optimizer):
        return optimizer.step()

    @staticmethod
    def update():
        return None


def build_grad_scaler(enabled: bool = True):
    if not enabled:
        return NoOpGradScaler()
    return torch.cuda.amp.GradScaler(enabled=True)


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
