from .crog import CROG
from .ssg import SSG
from loguru import logger


def build_crog(args):
    model = CROG(args)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    return model, param_list


def _build_drog_family(model_class, args):
    model = model_class(args)
    backbone = []
    head = []
    frozen = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(parameter)
        elif name.startswith(("txt_backbone", "dinov2")):
            backbone.append(parameter)
        else:
            head.append(parameter)
    logger.info(
        "{}: Backbone={}, Head={}, Frozen={}".format(
            model_class.__name__, len(backbone), len(head), len(frozen)
        )
    )
    param_list = [
        {
            "params": backbone,
            "initial_lr": args.lr_multi * args.base_lr,
        },
        {
            "params": head,
            "initial_lr": args.base_lr,
        },
    ]
    return model, param_list


def build_drog(args):
    from .drog import DROG

    return _build_drog_family(DROG, args)


def build_drogoff(args):
    from .drogoff import DROGOFF

    return _build_drog_family(DROGOFF, args)


def _build_toolrgs_family(args):
    from .toolrgs import build_toolrgs_model

    architecture = str(args.architecture).strip().lower()
    model = build_toolrgs_model(architecture, args)
    backbone = []
    head = []
    frozen = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(parameter)
        elif name.startswith(
            ("backbone", "bridger", "txt_backbone", "dinov2")
        ) and "positional_embedding" not in name:
            backbone.append(parameter)
        else:
            head.append(parameter)
    logger.info(
        "{}: Backbone={}, Head={}, Frozen={}".format(
            type(model).__name__,
            len(backbone),
            len(head),
            len(frozen),
        )
    )
    return model, [
        {
            "params": backbone,
            "lr": args.lr_multi * args.base_lr,
            "initial_lr": args.lr_multi * args.base_lr,
        },
        {
            "params": head,
            "lr": args.base_lr,
            "initial_lr": args.base_lr,
        },
    ]


def build_model(args):
    """Select an architecture without changing CROG's model or evaluator."""
    architecture = str(getattr(args, "architecture", "crog")).lower()
    builders = {
        "crog": build_crog,
        "drog": build_drog,
        "drogoff": build_drogoff,
    }
    if architecture in builders:
        return builders[architecture](args)
    toolrgs_models = {
        "crogoff",
        "etrg",
        "etrg_rgb",
        "ggcnnclip",
        "ggcnn_clip",
        "grconvnetclip",
        "grconvnet_clip",
        "graspmamba",
        "grasp_mamba",
        "lgd",
        "maplegrasp",
        "maple_grasp",
    }
    if architecture in toolrgs_models:
        return _build_toolrgs_family(args)
    choices = ", ".join(sorted(set(builders) | toolrgs_models))
    raise ValueError(
        f"Unknown MODEL.architecture {architecture!r}; choose one of: {choices}"
    )

def build_ssg(args):
    model = SSG(args)

    return model, model.parameters()
