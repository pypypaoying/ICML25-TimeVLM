#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TASK_PREFIXES = [
    "long_term_forecast",
    "few_shot_forecast",
    "zero_shot_forecast",
    "short_term_forecast",
    "imputation",
    "anomaly_detection",
    "classification",
]
HORIZONS = [96, 192, 336, 720]
METRIC_ALIASES = {
    "mse": ["test/mse", "mse"],
    "mae": ["test/mae", "mae"],
    "rmse": ["test/rmse", "rmse"],
    "mape": ["test/mape", "mape"],
    "mspe": ["test/mspe", "mspe"],
    "dtw": ["test/dtw", "dtw"],
}
DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "Traffic",
    "Weather",
]


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, decimals):
    value = _to_float(value)
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _metric(record, metric_name):
    metrics = record.get("metrics", {})
    for key in METRIC_ALIASES[metric_name]:
        if key in metrics:
            return _to_float(metrics[key])
    return None


def _stem(path_like):
    if not path_like:
        return ""
    return os.path.splitext(os.path.basename(str(path_like)))[0]


def _task_from_setting(setting):
    for task_name in TASK_PREFIXES:
        if setting.startswith(task_name + "_"):
            return task_name
    return ""


def _parse_setting(setting):
    task_name = _task_from_setting(setting)
    parsed = {
        "setting": setting,
        "task_name": task_name,
        "dataset": "",
        "data": "",
        "model": "",
        "model_id": "",
        "vlm_type": "",
        "seq_len": None,
        "label_len": None,
        "pred_len": None,
        "d_model": None,
        "percent": None,
        "seed": None,
    }

    prefix_len = len(task_name) + 1 if task_name else 0
    parts = setting[prefix_len:].split("_") if prefix_len else setting.split("_")
    ft_idx = next((i for i, part in enumerate(parts) if part.startswith("ft")), None)
    if ft_idx is not None and ft_idx >= 4:
        parsed["vlm_type"] = parts[0]
        parsed["model"] = parts[ft_idx - 2]
        parsed["data"] = parts[ft_idx - 1]
        parsed["model_id"] = "_".join(parts[1:ft_idx - 2])
        parsed["dataset"] = parts[1]

    patterns = {
        "seq_len": r"(?:^|_)sl(\d+)(?:_|$)",
        "label_len": r"(?:^|_)ll(\d+)(?:_|$)",
        "pred_len": r"(?:^|_)pl(\d+)(?:_|$)",
        "d_model": r"(?:^|_)dm(\d+)(?:_|$)",
        "percent": r"(?:^|_)fs([0-9.]+)(?:_|$)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, setting)
        if not match:
            continue
        parsed[key] = float(match.group(1)) if key == "percent" else int(match.group(1))
    return parsed


def _normalize_record(record, source_path):
    data_path = record.get("data_path", "")
    dataset = record.get("dataset") or _stem(data_path) or record.get("data", "")
    task_name = record.get("task_name", "")
    target_data = record.get("target_data", "")
    target_data_path = record.get("target_data_path", "")
    if task_name == "zero_shot_forecast":
        source = record.get("data", "") or dataset
        target = target_data or _stem(target_data_path)
        dataset_label = f"{source}->{target}" if target else dataset
    else:
        dataset_label = dataset

    normalized = dict(record)
    normalized["dataset"] = dataset
    normalized["dataset_label"] = dataset_label
    normalized["source_path"] = str(source_path)
    normalized["source_mtime"] = source_path.stat().st_mtime
    for metric_name in METRIC_ALIASES:
        normalized[metric_name] = _metric(record, metric_name)
    return normalized


def _record_from_metrics_npy(metrics_path):
    values = np.load(metrics_path)
    setting = metrics_path.parent.name
    parsed = _parse_setting(setting)
    metrics = {}
    for key, value in zip(["mae", "mse", "rmse", "mape", "mspe"], values):
        metrics[f"test/{key}"] = float(value)
    parsed["metrics"] = metrics
    parsed["result_dir"] = str(metrics_path.parent.resolve())
    return parsed


def load_records(results_dir):
    results_dir = Path(results_dir)
    records = []
    json_paths = set()

    for path in results_dir.glob("*/metrics.json"):
        json_paths.add(path.parent)
        with path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        records.append(_normalize_record(record, path))

    for path in results_dir.glob("*/metrics.npy"):
        if path.parent in json_paths:
            continue
        record = _record_from_metrics_npy(path)
        records.append(_normalize_record(record, path))

    return records


def _matches_filter(value, selected):
    if not selected:
        return True
    value = "" if value is None else str(value)
    return value.lower() in {item.lower() for item in selected}


def filter_records(records, tasks, datasets, percents):
    filtered = []
    percent_values = {str(item) for item in percents}
    for record in records:
        if not _matches_filter(record.get("task_name"), tasks):
            continue
        if datasets and not (
            _matches_filter(record.get("dataset"), datasets)
            or _matches_filter(record.get("dataset_label"), datasets)
        ):
            continue
        if percent_values:
            percent = record.get("percent")
            if str(percent) not in percent_values and _fmt(percent, 3).rstrip("0").rstrip(".") not in percent_values:
                continue
        if record.get("mse") is None and record.get("mae") is None:
            continue
        filtered.append(record)
    return filtered


def _dataset_sort_key(name):
    if name in DATASET_ORDER:
        return (0, DATASET_ORDER.index(name))
    return (1, name)


def latest_by_table_key(records):
    keyed = {}
    for record in records:
        key = (
            record.get("task_name", ""),
            str(record.get("percent", "")),
            record.get("dataset_label", ""),
            record.get("pred_len"),
        )
        current = keyed.get(key)
        if current is None or record.get("source_mtime", 0) >= current.get("source_mtime", 0):
            keyed[key] = record
    return list(keyed.values())


def write_raw_csv(records, output_path, method_name):
    fields = [
        "method",
        "task_name",
        "dataset",
        "dataset_label",
        "data",
        "target_data",
        "percent",
        "seq_len",
        "label_len",
        "pred_len",
        "model",
        "model_id",
        "d_model",
        "vlm_type",
        "use_mem_gate",
        "seed",
        "mse",
        "mae",
        "rmse",
        "mape",
        "mspe",
        "dtw",
        "setting",
        "result_dir",
        "source_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (
            item.get("task_name", ""),
            str(item.get("percent", "")),
            _dataset_sort_key(item.get("dataset_label", "")),
            item.get("pred_len") or 0,
        )):
            row = {field: record.get(field, "") for field in fields}
            row["method"] = method_name
            writer.writerow(row)


def _table_groups(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.get("task_name", ""), str(record.get("percent", "")))].append(record)
    return grouped


def _task_title(task_name, percent):
    if task_name == "long_term_forecast":
        return f"Long-Term Forecasting ({percent} data)"
    if task_name == "few_shot_forecast":
        return f"Few-Shot Forecasting ({percent} data)"
    if task_name == "zero_shot_forecast":
        return "Zero-Shot Transfer Forecasting"
    return task_name or "Forecasting"


def build_table_rows(records, method_name):
    table_rows = []
    for (task_name, percent), group in sorted(_table_groups(records).items()):
        by_dataset = defaultdict(dict)
        for record in group:
            by_dataset[record.get("dataset_label", "")][record.get("pred_len")] = record
        for dataset in sorted(by_dataset, key=_dataset_sort_key):
            horizon_map = by_dataset[dataset]
            for metric_name in ["mse", "mae"]:
                values = [horizon_map.get(h, {}).get(metric_name) for h in HORIZONS]
                present = [value for value in values if value is not None]
                table_rows.append({
                    "task_name": task_name,
                    "percent": percent,
                    "method": method_name,
                    "dataset": dataset,
                    "metric": metric_name.upper(),
                    **{str(h): values[i] for i, h in enumerate(HORIZONS)},
                    "avg": sum(present) / len(present) if present else None,
                })
    return table_rows


def write_table_csv(table_rows, output_path, decimals):
    fields = ["task_name", "percent", "method", "dataset", "metric"] + [str(h) for h in HORIZONS] + ["avg"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in table_rows:
            formatted = dict(row)
            for horizon in [str(h) for h in HORIZONS] + ["avg"]:
                formatted[horizon] = _fmt(formatted.get(horizon), decimals)
            writer.writerow(formatted)


def write_markdown(table_rows, output_path, method_name, decimals):
    lines = [
        f"# {method_name} Forecasting Results",
        "",
        "Values are generated from `results/*/metrics.json` or legacy `metrics.npy` files.",
        "",
    ]
    grouped = defaultdict(list)
    for row in table_rows:
        grouped[(row["task_name"], row["percent"])].append(row)

    for (task_name, percent), rows in sorted(grouped.items()):
        lines.append(f"## {_task_title(task_name, percent)}")
        lines.append("")
        lines.append("| Dataset | Metric | 96 | 192 | 336 | 720 | Avg |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            values = [_fmt(row.get(str(h)), decimals) for h in HORIZONS]
            lines.append(
                "| {dataset} | {metric} | {values} | {avg} |".format(
                    dataset=row["dataset"],
                    metric=row["metric"],
                    values=" | ".join(values),
                    avg=_fmt(row.get("avg"), decimals),
                )
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Time-VLM forecasting metrics into paper-style MSE/MAE tables."
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--method_name", default="TimeVLM")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--percents", nargs="*", default=[])
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--allow_empty", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_records(args.results_dir)
    records = filter_records(records, args.tasks, args.datasets, args.percents)
    records = latest_by_table_key(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not records:
        message = "No forecasting metric records found."
        if args.allow_empty:
            print(message)
            return 0
        raise SystemExit(message)

    raw_csv = output_dir / "forecasting_metrics_raw.csv"
    table_csv = output_dir / "forecasting_table.csv"
    table_md = output_dir / "forecasting_table.md"

    table_rows = build_table_rows(records, args.method_name)
    write_raw_csv(records, raw_csv, args.method_name)
    write_table_csv(table_rows, table_csv, args.decimals)
    write_markdown(table_rows, table_md, args.method_name, args.decimals)

    print(f"Wrote {raw_csv}")
    print(f"Wrote {table_csv}")
    print(f"Wrote {table_md}")


if __name__ == "__main__":
    main()
