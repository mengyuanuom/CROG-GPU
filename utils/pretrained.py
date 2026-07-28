"""Safe on-demand downloads for official ToolRGSNPU/CROG-NPU backbones."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Optional
import urllib.request


@dataclass(frozen=True)
class PretrainedArtifact:
    filename: str
    name: str
    url: str
    sha256: Optional[str] = None
    sha256_prefix: Optional[str] = None


ARTIFACTS = {
    "clip-rn50": PretrainedArtifact(
        "RN50.pt",
        "OpenAI CLIP RN50",
        "https://openaipublic.azureedge.net/clip/models/"
        "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/"
        "RN50.pt",
        sha256="afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
    ),
    "clip-rn101": PretrainedArtifact(
        "RN101.pt",
        "OpenAI CLIP RN101",
        "https://openaipublic.azureedge.net/clip/models/"
        "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/"
        "RN101.pt",
        sha256="8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599",
    ),
    "clip-vit-b16": PretrainedArtifact(
        "ViT-B-16.pt",
        "OpenAI CLIP ViT-B/16",
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
        "ViT-B-16.pt",
        sha256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    ),
    "dinov2-vitb14-reg4": PretrainedArtifact(
        "dinov2_vitb14_reg4_pretrain.pth",
        "Meta DINOv2 ViT-B/14 with registers",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
        "dinov2_vitb14_reg4_pretrain.pth",
    ),
    "mambavision-t": PretrainedArtifact(
        "mambavision_tiny_1k.pth.tar",
        "NVIDIA MambaVision-T",
        "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/"
        "mambavision_tiny_1k.pth.tar",
        sha256="952a3e486f94bbe863c753a7ecabe282b2e3b8adbb0d98057e047e4f554c2a9b",
    ),
    "resnet18": PretrainedArtifact(
        "resnet18-f37072fd.pth",
        "PyTorch ResNet-18",
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        sha256_prefix="f37072fd",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(path: Path, artifact: PretrainedArtifact) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if artifact.sha256 is None and artifact.sha256_prefix is None:
        return True
    digest = _sha256(path)
    if artifact.sha256 is not None:
        return digest == artifact.sha256
    return digest.startswith(artifact.sha256_prefix or "")


def _artifact_key_for_path(path: Path) -> Optional[str]:
    for key, artifact in ARTIFACTS.items():
        if artifact.filename == path.name:
            return key
    return None


def _acquire_lock(lock_path: Path, target: Path, artifact: PretrainedArtifact, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        if _valid(target, artifact):
            return None
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for another process to download {target}"
                )
            time.sleep(1.0)


def ensure_pretrained(path, artifact_key: Optional[str] = None, *, lock_timeout=1800.0) -> Path:
    """Return a valid checkpoint, downloading the official file only if absent."""
    target = Path(path).expanduser()
    artifact_key = artifact_key or _artifact_key_for_path(target)
    if artifact_key is None:
        if target.is_file() and target.stat().st_size:
            return target
        known = ", ".join(sorted(item.filename for item in ARTIFACTS.values()))
        raise FileNotFoundError(
            f"No automatic download is registered for {target}. "
            f"Known official filenames: {known}"
        )
    artifact = ARTIFACTS[artifact_key]
    if _valid(target, artifact):
        print(f"[weights] found: {target}", flush=True)
        return target
    if target.exists():
        raise RuntimeError(
            f"Existing checkpoint is empty or invalid: {target}. "
            f"Remove it to download the official file again:\n{artifact.url}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    lock_fd = _acquire_lock(lock_path, target, artifact, float(lock_timeout))
    if lock_fd is None:
        return target
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        if _valid(target, artifact):
            return target

        temporary = target.with_name(f"{target.name}.part.{os.getpid()}")
        request = urllib.request.Request(
            artifact.url,
            headers={"User-Agent": "CROG-NPU-weight-downloader/1.0"},
        )
        print(
            f"[weights] missing; downloading {artifact.name}\n"
            f"  from: {artifact.url}\n"
            f"  to:   {target}",
            flush=True,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
            if not _valid(temporary, artifact):
                raise RuntimeError(
                    f"Downloaded checkpoint failed validation: {temporary}"
                )
            os.replace(temporary, target)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {artifact.name} from:\n{artifact.url}\n"
                f"Download it manually to {target}. Original error: {exc}"
            ) from exc
        print(f"[weights] ready: {target}", flush=True)
        return target
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)