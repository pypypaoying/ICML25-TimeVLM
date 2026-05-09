import datetime as _datetime
import json
import os

import numpy as np


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _dataset_name(args):
    data_path = getattr(args, "data_path", "")
    if data_path:
        return os.path.splitext(os.path.basename(data_path))[0]
    return getattr(args, "data", "")


def save_forecast_metrics(args, setting, metrics, result_dir, extra=None):
    """Write one structured forecasting metric record for table aggregation."""
    os.makedirs(result_dir, exist_ok=True)
    result_dir = os.path.abspath(result_dir)

    payload = {
        "created_at": _datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "setting": setting,
        "task_name": getattr(args, "task_name", ""),
        "model": getattr(args, "model", ""),
        "model_id": getattr(args, "model_id", ""),
        "dataset": _dataset_name(args),
        "data": getattr(args, "data", ""),
        "data_path": getattr(args, "data_path", ""),
        "target_data": getattr(args, "target_data", ""),
        "target_data_path": getattr(args, "target_data_path", ""),
        "features": getattr(args, "features", ""),
        "seq_len": getattr(args, "seq_len", None),
        "label_len": getattr(args, "label_len", None),
        "pred_len": getattr(args, "pred_len", None),
        "percent": getattr(args, "percent", None),
        "seed": getattr(args, "seed", None),
        "d_model": getattr(args, "d_model", None),
        "vlm_type": getattr(args, "vlm_type", ""),
        "ablation_variant": getattr(args, "ablation_variant", "full"),
        "use_mem_gate": getattr(args, "use_mem_gate", None),
        "finetune_vlm": getattr(args, "finetune_vlm", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "batch_size": getattr(args, "batch_size", None),
        "train_epochs": getattr(args, "train_epochs", None),
        "metrics": _jsonable(metrics),
        "result_dir": result_dir,
    }
    if extra:
        payload.update(_jsonable(extra))

    record_path = os.path.join(result_dir, "metrics.json")
    payload["record_path"] = record_path
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    index_path = os.path.join(os.path.dirname(result_dir), "forecasting_runs.jsonl")
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")

    return payload
