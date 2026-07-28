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


def build_model(args):
    """Select an architecture without changing CROG's model or evaluator."""
    architecture = str(getattr(args, "architecture", "crog")).lower()
    builders = {
        "crog": build_crog,
        "drog": build_drog,
        "drogoff": build_drogoff,
    }
    if architecture not in builders:
        choices = ", ".join(sorted(builders))
        raise ValueError(
            f"Unknown MODEL.architecture {architecture!r}; choose one of: {choices}"
        )
    return builders[architecture](args)

def build_ssg(args):
    model = SSG(args)

    return model, model.parameters()
