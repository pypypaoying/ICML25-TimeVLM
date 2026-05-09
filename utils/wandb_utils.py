import numbers


def is_wandb_enabled(args):
    return bool(getattr(args, "use_wandb", False))


def _safe_config(args, setting):
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool, type(None))):
            config[key] = value
        elif isinstance(value, (list, tuple)):
            config[key] = list(value)
        else:
            config[key] = str(value)
    config["setting"] = setting
    return config


def _clean_metrics(metrics):
    clean = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, numbers.Number):
            clean[key] = float(value)
        else:
            clean[key] = value
    return clean


def init_wandb(args, setting):
    if not is_wandb_enabled(args):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB logging was requested, but the 'wandb' package is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    tags = [
        tag.strip()
        for tag in getattr(args, "wandb_tags", "").split(",")
        if tag.strip()
    ]
    run_name = getattr(args, "wandb_run_name", "") or setting
    group = getattr(args, "wandb_group", "") or "{}-{}-{}p".format(
        args.data, args.task_name, args.percent
    )
    entity = getattr(args, "wandb_entity", "") or None

    run = wandb.init(
        project=getattr(args, "wandb_project", "time-vlm-minimal-reproduction"),
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        config=_safe_config(args, setting),
        mode=getattr(args, "wandb_mode", "online"),
    )

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("test_proxy/*", step_metric="epoch")
    wandb.define_metric("test/*")

    return run


def wandb_log(args, metrics, step=None):
    if not is_wandb_enabled(args):
        return

    import wandb

    if wandb.run is None:
        return

    data = _clean_metrics(metrics)
    if step is not None:
        data["epoch"] = step
    wandb.log(data)


def finish_wandb(args):
    if not is_wandb_enabled(args):
        return

    import wandb

    if wandb.run is not None:
        wandb.finish()
