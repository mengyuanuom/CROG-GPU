"""ToolRGS model variants isolated from the original CROG implementation."""

from importlib import import_module


MODEL_REGISTRY = {
    "etrg": ("etrg", "ETRG"),
    "etrg_rgb": ("etrg", "ETRG"),
    "crogoff": ("crogoff", "CROGOFF"),
    "ggcnnclip": ("ggcnnclip", "GGCNN_CLIP"),
    "ggcnn_clip": ("ggcnnclip", "GGCNN_CLIP"),
    "grconvnetclip": ("grconvnetclip", "GenerativeResnet_CLIP"),
    "grconvnet_clip": ("grconvnetclip", "GenerativeResnet_CLIP"),
    "graspmamba": ("graspmamba", "GraspMamba"),
    "grasp_mamba": ("graspmamba", "GraspMamba"),
    "lgd": ("lgd", "LGD"),
    "maplegrasp": ("maplegrasp", "MapleGrasp"),
    "maple_grasp": ("maplegrasp", "MapleGrasp"),
}


def build_toolrgs_model(name, cfg):
    normalized = str(name).strip().lower()
    try:
        module_name, class_name = MODEL_REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown ToolRGS model {name!r}; available: {available}"
        ) from exc
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, class_name)(cfg)
