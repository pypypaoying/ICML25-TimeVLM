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
    wandb.define_metric("paper_table/*")

    return run


def _metric_value(metrics, *keys):
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, numbers.Number):
            return float(value)
    return None


def _record_dataset_label(record):
    if record.get("task_name") == "zero_shot_forecast":
        source = record.get("data") or record.get("dataset", "")
        target = record.get("target_data") or record.get("target_data_path", "")
        if target and "." in str(target):
            target = str(target).rsplit(".", 1)[0]
        if source and target:
            return "{}->{}".format(source, target)
    return record.get("dataset") or record.get("data") or ""


def wandb_log_forecast_summary(args, record):
    if not is_wandb_enabled(args):
        return

    import wandb

    if wandb.run is None:
        return

    metrics = _clean_metrics(record.get("metrics", {}))
    metric_values = {
        "mse": _metric_value(metrics, "test/mse", "mse"),
        "mae": _metric_value(metrics, "test/mae", "mae"),
        "rmse": _metric_value(metrics, "test/rmse", "rmse"),
        "mape": _metric_value(metrics, "test/mape", "mape"),
        "mspe": _metric_value(metrics, "test/mspe", "mspe"),
        "dtw": _metric_value(metrics, "test/dtw", "dtw"),
    }

    summary_fields = {
        "paper_table/task_name": record.get("task_name", ""),
        "paper_table/dataset": _record_dataset_label(record),
        "paper_table/percent": record.get("percent"),
        "paper_table/seq_len": record.get("seq_len"),
        "paper_table/pred_len": record.get("pred_len"),
        "paper_table/seed": record.get("seed"),
    }
    for key, value in summary_fields.items():
        wandb.run.summary[key] = value

    scalar_log = {}
    for metric_name, value in metric_values.items():
        if value is None:
            continue
        wandb.run.summary["final/{}".format(metric_name)] = value
        wandb.run.summary["paper_table/{}".format(metric_name)] = value
        scalar_log["paper_table/{}".format(metric_name)] = value

    if metric_values["mse"] is not None:
        wandb.run.summary["MSE"] = metric_values["mse"]
    if metric_values["mae"] is not None:
        wandb.run.summary["MAE"] = metric_values["mae"]

    table = wandb.Table(
        columns=[
            "task_name",
            "dataset",
            "percent",
            "seq_len",
            "pred_len",
            "mse",
            "mae",
            "rmse",
            "mape",
            "mspe",
            "dtw",
            "setting",
        ],
        data=[[
            record.get("task_name", ""),
            _record_dataset_label(record),
            record.get("percent"),
            record.get("seq_len"),
            record.get("pred_len"),
            metric_values["mse"],
            metric_values["mae"],
            metric_values["rmse"],
            metric_values["mape"],
            metric_values["mspe"],
            metric_values["dtw"],
            record.get("setting", ""),
        ]],
    )
    scalar_log["paper_table/mse_mae"] = table
    wandb.log(scalar_log)


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
