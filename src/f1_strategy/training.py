from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any

from f1_strategy.artifacts import DEFAULT_ARTIFACT_ROOT, create_model_artifact_bundle
from f1_strategy.config import Settings, load_settings
from f1_strategy.domain import TireCompound
from f1_strategy.engine import InferenceEngine
from f1_strategy.evaluation import run_evaluation
from f1_strategy.feature_store import OnlineFeatureStore
from f1_strategy.models import (
    FEATURE_NAMES,
    ModelConfig,
    create_serving_model,
    feature_schema_hash,
    features_to_vector,
    model_manifest_path,
    write_model_manifest,
)
from f1_strategy.replay import (
    DEFAULT_REPLAY_DATASET,
    replay_reference_lap_time_s,
    run_benchmark_replay_suite,
    run_replay_evaluation,
)
from f1_strategy.serialization import telemetry_from_dict
from f1_strategy.simulation import RaceSimulator, SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and export a tabular serving model.")
    parser.add_argument(
        "--backend",
        choices=["xgboost", "lightgbm", "catboost", "sequence"],
        default="xgboost",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--laps", type=int, default=28)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=140)
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Optional artifact registry root for a train/evaluate bundle.",
    )
    parser.add_argument(
        "--replay-dataset",
        default=str(DEFAULT_REPLAY_DATASET),
        help="Replay telemetry dataset used as the holdout promotion gate.",
    )
    parser.add_argument(
        "--real-data",
        nargs="*",
        default=None,
        metavar="PATH",
        help="One or more CSV/JSONL replay files to augment synthetic training rows with real lap times.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        default=False,
        help="Log training metrics to MLflow (requires pip install mlflow).",
    )
    return parser


_REQUIRED_CSV_COLUMNS = {
    "session_id", "car_id", "lap", "sector",
    "speed_kph", "throttle", "brake", "compound",
    "tire_temp_fl", "tire_temp_fr", "tire_temp_rl", "tire_temp_rr",
    "ers_soc", "fuel_kg", "track_temp_c",
}

_COLUMN_RANGES: dict[str, tuple[float, float]] = {
    "throttle": (0.0, 1.0),
    "brake": (0.0, 1.0),
    "ers_soc": (0.0, 1.0),
    "lateral_g": (0.0, 10.0),
    "speed_kph": (0.0, 400.0),
    "tire_temp_fl": (0.0, 250.0),
    "tire_temp_fr": (0.0, 250.0),
    "tire_temp_rl": (0.0, 250.0),
    "tire_temp_rr": (0.0, 250.0),
    "track_temp_c": (-10.0, 80.0),
    "fuel_kg": (0.0, 200.0),
}


def _validate_csv_schema(records: list[dict], path: str) -> None:
    """Raise ValueError with a diagnostic message if data quality checks fail.

    Validates:
    - All required columns are present in the header.
    - Key numeric columns are within plausible physical ranges.
    - Missing-value rate for required columns is below 50%.
    """
    if not records:
        return
    header = set(records[0].keys())
    missing_cols = _REQUIRED_CSV_COLUMNS - header
    if missing_cols:
        raise ValueError(
            f"Data quality failure in {path}: missing required columns {sorted(missing_cols)}"
        )

    n = len(records)
    for col, (lo, hi) in _COLUMN_RANGES.items():
        if col not in header:
            continue
        violations = []
        blank = 0
        for row in records:
            v = row.get(col, "")
            if v in ("", None, "None", "null", "nan", "NaN"):
                blank += 1
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                blank += 1
                continue
            if not (lo <= fv <= hi):
                violations.append(fv)

        blank_rate = blank / n
        if blank_rate > 0.50:
            raise ValueError(
                f"Data quality failure in {path}: column '{col}' has {blank_rate:.0%} "
                f"missing values (threshold 50%)"
            )
        if len(violations) > 0.05 * n:
            raise ValueError(
                f"Data quality failure in {path}: column '{col}' has "
                f"{len(violations)} rows outside [{lo}, {hi}] "
                f"(>{5:.0f}% of {n} records). Sample: {violations[:3]}"
            )


def _coerce_csv_row(row: dict) -> dict:
    """Cast CSV string values to the types TelemetryEvent expects."""
    _INT_FIELDS = {"lap", "sector", "timestamp_ms"}
    _FLOAT_FIELDS = {
        "speed_kph", "throttle", "brake", "steering_angle",
        "tire_temp_fl", "tire_temp_fr", "tire_temp_rl", "tire_temp_rr",
        "brake_temp", "slip_angle", "lateral_g", "ers_soc",
        "ers_deployment_kw", "fuel_kg", "track_temp_c", "air_temp_c", "humidity",
    }
    out: dict = {}
    for k, v in row.items():
        if v in ("", None):
            continue
        if k in _INT_FIELDS:
            out[k] = int(float(v))
        elif k in _FLOAT_FIELDS:
            out[k] = float(v)
        elif k == "lap_time_s":
            if v not in ("None", "null"):
                out[k] = float(v)
        else:
            out[k] = v
    return out


def _lap_time_bounds(records: list[dict]) -> tuple[float, float]:
    """Return (lo, hi) bounds using median ± 5×MAD to reject SC/red-flag laps."""
    from statistics import median as _median
    times = [
        float(r["lap_time_s"])
        for r in records
        if r.get("lap_time_s") not in ("", "None", "null", None)
    ]
    if not times:
        return (0.0, float("inf"))
    med = _median(times)
    mad = _median([abs(t - med) for t in times]) or 1.0
    return (med - 5.0 * mad, med + 5.0 * mad)


def _manifest_reference_base(path: str) -> float | None:
    """Return the manifest's reference_lap_time_s for this dataset, or None if unavailable."""
    import math
    manifest_path = Path(path).with_suffix(Path(path).suffix + ".manifest.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = manifest.get("reference_lap_time_s")
        if value is not None:
            fval = float(value)
            if math.isfinite(fval):
                return fval
    except (OSError, TypeError, ValueError):
        pass
    return None


def _real_data_rows(
    path: str,
    base_lap_time_s: float | None = None,
) -> tuple[list[list[float]], list[float]]:
    """Convert a labeled replay CSV/JSONL into (feature_vectors, targets).

    Base priority (highest first):
    1. Explicit base_lap_time_s argument
    2. manifest reference_lap_time_s (matches what replay evaluation uses)
    3. Filtered median of this file's lap times

    Using the manifest reference ensures training targets align with the
    evaluation's actual_delta computation for zero systematic offset.
    """
    settings = load_settings()
    rows: list[list[float]] = []
    targets: list[float] = []
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        import json as _json
        records: list[dict] = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(_json.loads(line))
    else:
        with p.open(encoding="utf-8", newline="") as fh:
            records = list(csv.DictReader(fh))

    _validate_csv_schema(records, path)
    lo, hi = _lap_time_bounds(records)

    # First pass: collect outlier-filtered records for auto-base computation.
    valid_records: list[tuple[dict, float]] = []
    for record in records:
        raw = record.get("lap_time_s")
        if not raw or raw in ("", "None", "null", "nan", "NaN"):
            continue
        try:
            t = float(raw)
        except ValueError:
            continue
        if lo <= t <= hi:
            valid_records.append((record, t))

    # Determine base: explicit > manifest reference > filtered median.
    if base_lap_time_s is not None:
        effective_base = base_lap_time_s
    else:
        manifest_base = _manifest_reference_base(path)
        if manifest_base is not None:
            effective_base = manifest_base
        else:
            from statistics import median as _median
            file_times = [t for _, t in valid_records]
            effective_base = _median(file_times) if file_times else settings.base_lap_time_s

    # Store is created after effective_base is known so circuit_base_lap_s is set correctly.
    store = OnlineFeatureStore(window_size=settings.feature_window_size, base_lap_time_s=effective_base)

    extra_skipped = 0
    for record, lap_time_val in valid_records:
        try:
            coerced = _coerce_csv_row(record)
            # actual_tire_age_laps is an OpenF1-only column not in TelemetryEvent;
            # extract it before parsing so it doesn't crash telemetry_from_dict.
            raw_age = coerced.pop("actual_tire_age_laps", None)
            event = telemetry_from_dict(coerced)
            features = store.ingest(event)
            if raw_age not in (None, "", "nan", "NaN"):
                try:
                    from dataclasses import replace as _dc_replace
                    features = _dc_replace(features, tire_age_laps=int(float(raw_age)))
                except (ValueError, TypeError):
                    pass
            rows.append(features_to_vector(features))
            targets.append(lap_time_val - effective_base)
        except Exception:
            extra_skipped += 1

    total_skipped = (len(records) - len(valid_records)) + extra_skipped
    if total_skipped:
        print(f"  skipped {total_skipped} outlier/malformed rows in {path} (base={effective_base:.1f}s)")
    else:
        print(f"  loaded {len(rows)} rows from {path} (base={effective_base:.1f}s)")
    return rows, targets


def _synthetic_rows_for_base(
    laps: int,
    seeds: int,
    base: float,
    settings: Settings,
    seed_offset: int = 0,
) -> tuple[list[list[float]], list[float]]:
    """Generate synthetic simulation rows at a specific base lap time."""
    rows: list[list[float]] = []
    targets: list[float] = []
    compounds = list(TireCompound)
    for i in range(1, seeds + 1):
        seed = i + seed_offset
        compound = compounds[(seed - 1) % len(compounds)]
        simulator = RaceSimulator(
            SimulationConfig(
                session_id=f"train-{seed}",
                car_id=f"car-{seed % 20:02d}",
                laps=laps,
                seed=seed,
                compound=compound,
                base_lap_time_s=base,
            )
        )
        store = OnlineFeatureStore(window_size=settings.feature_window_size, base_lap_time_s=base)
        for event in simulator.events():
            features = store.ingest(event)
            if event.lap_time_s is None:
                continue
            rows.append(features_to_vector(features))
            targets.append(event.lap_time_s - base)
    return rows, targets


def _training_rows(
    laps: int,
    seeds: int,
    real_data_paths: list[str] | None = None,
    base_lap_time_s: float | None = None,
) -> tuple[list[list[float]], list[float]]:
    settings = load_settings()
    effective_base = base_lap_time_s if base_lap_time_s is not None else settings.base_lap_time_s
    rows: list[list[float]] = []
    targets: list[float] = []

    if not real_data_paths:
        # Synthetic-only: all rows at the global base.
        s_rows, s_targets = _synthetic_rows_for_base(laps, seeds, effective_base, settings)
        return s_rows, s_targets

    # Real-data mode: generate per-circuit synthetic+real pairs so that every
    # circuit's synthetic and real targets are always on the same normalised scale.
    # No global base=90 synthetic batch — instead each circuit gets its own seeds
    # so the circuit_base_lap_s feature is accurate for every row in training.
    seeds_per_circuit = max(8, seeds // len(real_data_paths))
    total_real = 0
    for idx, path in enumerate(real_data_paths):
        real_rows, real_targets = _real_data_rows(path, base_lap_time_s=base_lap_time_s)
        if not real_rows:
            continue
        # Infer the circuit base that _real_data_rows actually used.
        # Uses same priority: explicit > manifest > filtered median.
        if base_lap_time_s is not None:
            circuit_base = base_lap_time_s
        else:
            manifest_base = _manifest_reference_base(path)
            if manifest_base is not None:
                circuit_base = manifest_base
            else:
                from statistics import median as _median
                import csv as _csv
                with Path(path).open(encoding="utf-8", newline="") as fh:
                    records = list(_csv.DictReader(fh))
                lo, hi = _lap_time_bounds(records)
                times = [
                    float(r["lap_time_s"]) for r in records
                    if r.get("lap_time_s") not in ("", "None", "null", "nan", "NaN")
                    and lo <= float(r["lap_time_s"]) <= hi
                ]
                circuit_base = _median(times) if times else effective_base

        synth_rows, synth_targets = _synthetic_rows_for_base(
            laps, seeds_per_circuit, circuit_base, settings,
            seed_offset=idx * seeds_per_circuit,
        )
        rows.extend(synth_rows)
        targets.extend(synth_targets)
        rows.extend(real_rows)
        targets.extend(real_targets)
        total_real += len(real_rows)
        print(f"  circuit {Path(path).stem}: {len(synth_rows)} synthetic + {len(real_rows)} real rows (base={circuit_base:.1f}s)")

    if total_real:
        print(f"  total: {len(rows)} rows ({total_real} real, {len(rows)-total_real} synthetic) across {len(real_data_paths)} circuit(s)")
    return rows, targets


def _default_output(backend: str) -> str:
    return {
        "xgboost": "models/xgboost_lap_delta.json",
        "lightgbm": "models/lightgbm_lap_delta.txt",
        "catboost": "models/catboost_lap_delta.cbm",
        "sequence": "models/sequence_lap_delta.pt",
    }[backend]


def train_xgboost_model(
    output: str,
    laps: int,
    seeds: int,
    rounds: int,
    real_data_paths: list[str] | None = None,
    use_mlflow: bool = False,
    base_lap_time_s: float | None = None,
) -> Path:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError('Install ML dependencies first: pip install -e ".[ml]"') from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds, real_data_paths=real_data_paths, base_lap_time_s=base_lap_time_s)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,
        "eta": 0.055,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 2.0,
        "lambda": 1.6,
        "alpha": 0.08,
        "seed": 7,
    }

    _mlflow_start(use_mlflow, "xgboost", params, len(rows), real_data_paths)
    matrix = xgb.DMatrix(rows, label=targets, feature_names=FEATURE_NAMES)
    booster = xgb.train(params=params, dtrain=matrix, num_boost_round=rounds)
    booster.set_attr(
        model_type="lap_delta_regressor",
        feature_schema=",".join(FEATURE_NAMES),
        feature_schema_hash=feature_schema_hash(),
        training_rows=str(len(rows)),
    )
    _mlflow_log_metrics(
        use_mlflow, rows, targets,
        lambda row: booster.predict(xgb.DMatrix([row], feature_names=FEATURE_NAMES))[0],
    )
    importance = _compute_feature_importance(booster, "xgboost")
    _mlflow_log_feature_importance(use_mlflow, importance)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    _save_feature_importance(output_path, importance)
    write_model_manifest(output_path, backend="xgboost", training_rows=len(rows))
    _mlflow_log_artifact(use_mlflow, output_path)
    return output_path


def train_lightgbm_model(
    output: str,
    laps: int,
    seeds: int,
    rounds: int,
    real_data_paths: list[str] | None = None,
    use_mlflow: bool = False,
    base_lap_time_s: float | None = None,
) -> Path:
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Install LightGBM dependencies first: pip install -e ".[ml]"') from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds, real_data_paths=real_data_paths, base_lap_time_s=base_lap_time_s)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.055,
        "num_leaves": 24,
        "min_data_in_leaf": 8,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.08,
        "lambda_l2": 1.6,
        "verbosity": -1,
        "seed": 7,
    }
    _mlflow_start(use_mlflow, "lightgbm", params, len(rows), real_data_paths)
    dataset = lgb.Dataset(
        np.asarray(rows, dtype="float32"),
        label=np.asarray(targets, dtype="float32"),
        feature_name=FEATURE_NAMES,
    )
    booster = lgb.train(params=params, train_set=dataset, num_boost_round=rounds)
    _mlflow_log_metrics(
        use_mlflow, rows, targets,
        lambda row: float(booster.predict([row])[0]),
    )
    importance = _compute_feature_importance(booster, "lightgbm")
    _mlflow_log_feature_importance(use_mlflow, importance)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    _save_feature_importance(output_path, importance)
    write_model_manifest(output_path, backend="lightgbm", training_rows=len(rows))
    _mlflow_log_artifact(use_mlflow, output_path)
    return output_path


def train_catboost_model(
    output: str,
    laps: int,
    seeds: int,
    rounds: int,
    real_data_paths: list[str] | None = None,
    use_mlflow: bool = False,
    base_lap_time_s: float | None = None,
) -> Path:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise RuntimeError("Install CatBoost first: pip install -e '.[catboost]'") from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds, real_data_paths=real_data_paths, base_lap_time_s=base_lap_time_s)
    catboost_params = {
        "iterations": rounds,
        "depth": 5,
        "learning_rate": 0.055,
        "loss_function": "RMSE",
        "random_seed": 7,
        "verbose": False,
    }
    _mlflow_start(use_mlflow, "catboost", catboost_params, len(rows), real_data_paths)
    model = CatBoostRegressor(**catboost_params)
    model.fit(Pool(rows, targets, feature_names=FEATURE_NAMES))
    _mlflow_log_metrics(
        use_mlflow, rows, targets,
        lambda row: float(model.predict([row])[0]),
    )
    importance = _compute_feature_importance(model, "catboost")
    _mlflow_log_feature_importance(use_mlflow, importance)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    _save_feature_importance(output_path, importance)
    write_model_manifest(output_path, backend="catboost", training_rows=len(rows))
    _mlflow_log_artifact(use_mlflow, output_path)
    return output_path


def train_sequence_model(
    output: str,
    laps: int,
    seeds: int,
    rounds: int,
    real_data_paths: list[str] | None = None,
    use_mlflow: bool = False,
    base_lap_time_s: float | None = None,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install Torch first: pip install -e '.[deep]'") from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds, real_data_paths=real_data_paths, base_lap_time_s=base_lap_time_s)
    seq_params = {"hidden_size": 24, "num_layers": 1, "lr": 0.025, "weight_decay": 0.001}
    _mlflow_start(use_mlflow, "sequence", seq_params, len(rows), real_data_paths)
    x_train = torch.tensor(rows, dtype=torch.float32)
    y_train = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    torch.manual_seed(7)

    class SequenceRegressor(torch.nn.Module):
        def __init__(self, feature_count: int) -> None:
            super().__init__()
            self.encoder = torch.nn.LSTM(
                input_size=feature_count,
                hidden_size=24,
                num_layers=1,
                batch_first=True,
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(24, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            sequence = x.unsqueeze(1)
            encoded, _ = self.encoder(sequence)
            return self.head(encoded[:, -1, :])

    model = SequenceRegressor(feature_count=len(FEATURE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.001)
    loss_fn = torch.nn.MSELoss()
    for _ in range(rounds):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    traced = torch.jit.trace(model, x_train[:1])
    traced.save(str(output_path))
    write_model_manifest(output_path, backend="sequence-torch", training_rows=len(rows))
    _mlflow_log_artifact(use_mlflow, output_path)
    return output_path


def train_model(
    backend: str,
    output: str,
    laps: int,
    seeds: int,
    rounds: int,
    real_data_paths: list[str] | None = None,
    use_mlflow: bool = False,
    base_lap_time_s: float | None = None,
) -> Path:
    if backend == "xgboost":
        return train_xgboost_model(output, laps, seeds, rounds, real_data_paths, use_mlflow, base_lap_time_s)
    if backend == "lightgbm":
        return train_lightgbm_model(output, laps, seeds, rounds, real_data_paths, use_mlflow, base_lap_time_s)
    if backend == "catboost":
        return train_catboost_model(output, laps, seeds, rounds, real_data_paths, use_mlflow, base_lap_time_s)
    if backend == "sequence":
        return train_sequence_model(output, laps, seeds, rounds, real_data_paths, use_mlflow, base_lap_time_s)
    raise ValueError(f"Unsupported training backend: {backend}")


def _mlflow_start(
    enabled: bool,
    backend: str,
    params: dict,
    training_rows: int,
    real_data_paths: list[str] | None,
) -> None:
    if not enabled:
        return
    try:
        import mlflow
        tracking_uri = load_settings().mlflow_tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = f"f1-tire-energy-{backend}"
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_param("backend", backend)
        mlflow.log_param("training_rows", training_rows)
        mlflow.log_param("real_data_paths", ",".join(real_data_paths) if real_data_paths else "synthetic")
        mlflow.log_param("feature_schema_version", "online-features-v5")
        for k, v in params.items():
            mlflow.log_param(k, v)
    except ImportError:
        print("MLflow not installed — skipping experiment tracking. pip install mlflow")
    except Exception as exc:
        print(f"MLflow logging failed: {exc}")


def _mlflow_log_metrics(
    enabled: bool,
    rows: list[list[float]],
    targets: list[float],
    predict_fn: object,
) -> None:
    """Compute and log training-set RMSE and MAE to the active MLflow run."""
    if not enabled:
        return
    try:
        import mlflow
        import math

        preds = [predict_fn(row) for row in rows]  # type: ignore[operator]
        n = len(targets)
        rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, targets)) / max(n, 1))
        mae = sum(abs(p - t) for p, t in zip(preds, targets)) / max(n, 1)
        mlflow.log_metric("train_rmse_lap_delta_s", rmse)
        mlflow.log_metric("train_mae_lap_delta_s", mae)
    except Exception:
        pass


def _compute_feature_importance(model: Any, backend: str) -> dict[str, float]:
    """Return {feature_name: gain_score} from the trained booster."""
    try:
        if backend == "xgboost":
            raw = model.get_score(importance_type="gain")
            return {k: float(v) for k, v in raw.items()}
        if backend == "lightgbm":
            scores = model.feature_importance(importance_type="gain")
            names = model.feature_name()
            return {name: float(score) for name, score in zip(names, scores)}
        if backend == "catboost":
            scores = model.get_feature_importance()
            names = model.feature_names_
            return {name: float(score) for name, score in zip(names, scores)}
    except Exception:
        pass
    return {}


def _save_feature_importance(output_path: Path, importance: dict[str, float]) -> None:
    if not importance:
        return
    importance_path = output_path.with_name(output_path.name + ".importance.json")
    importance_path.write_text(
        json.dumps(importance, sort_keys=True, indent=2), encoding="utf-8"
    )


def _mlflow_log_feature_importance(enabled: bool, importance: dict[str, float]) -> None:
    if not enabled or not importance:
        return
    try:
        import mlflow
        total = sum(importance.values()) or 1.0
        for name, score in sorted(importance.items(), key=lambda x: -x[1])[:30]:
            mlflow.log_metric(f"fi_{name}", round(score / total, 6))
    except Exception:
        pass


def _mlflow_log_artifact(enabled: bool, path: Path) -> None:
    if not enabled:
        return
    try:
        import mlflow
        mlflow.log_artifact(str(path))
        importance_path = path.with_name(path.name + ".importance.json")
        if importance_path.exists():
            mlflow.log_artifact(str(importance_path))
        mlflow.end_run()
    except Exception:
        pass


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or _default_output(args.backend)
    output_path = train_model(
        backend=args.backend,
        output=output,
        laps=args.laps,
        seeds=args.seeds,
        rounds=args.rounds,
        real_data_paths=args.real_data or None,
        use_mlflow=args.mlflow,
    )
    print(f"saved {args.backend} model: {output_path}")
    if args.artifact_root is not None:
        training_config = _training_config(
            backend=args.backend,
            output_path=output_path,
            laps=args.laps,
            seeds=args.seeds,
            rounds=args.rounds,
            real_data_paths=args.real_data or None,
            replay_dataset_path=args.replay_dataset,
        )
        report = run_evaluation(
            model_backend=args.backend,
            model_paths={args.backend: str(output_path), "sequence": str(output_path)},
        )
        _backend = args.backend
        _output = str(output_path)
        _replay_base = replay_reference_lap_time_s(args.replay_dataset)
        replay_report = run_replay_evaluation(
            args.replay_dataset,
            engine=InferenceEngine(
                model=create_serving_model(
                    config=ModelConfig(base_lap_time_s=_replay_base),
                    backend=_backend,
                    xgboost_model_path=_output if _backend == "xgboost" else "models/xgboost_lap_delta.json",
                    lightgbm_model_path=_output if _backend == "lightgbm" else "models/lightgbm_lap_delta.txt",
                    catboost_model_path=_output if _backend == "catboost" else "models/catboost_lap_delta.cbm",
                    sequence_model_path=_output if _backend == "sequence" else "models/sequence_lap_delta.pt",
                ),
                settings=_replace(load_settings(), base_lap_time_s=_replay_base),
            ),
        )
        replay_suite = run_benchmark_replay_suite(
            engine_factory=lambda base: InferenceEngine(
                model=create_serving_model(
                    config=ModelConfig(base_lap_time_s=base),
                    backend=_backend,
                    xgboost_model_path=_output if _backend == "xgboost" else "models/xgboost_lap_delta.json",
                    lightgbm_model_path=_output if _backend == "lightgbm" else "models/lightgbm_lap_delta.txt",
                    catboost_model_path=_output if _backend == "catboost" else "models/catboost_lap_delta.cbm",
                    sequence_model_path=_output if _backend == "sequence" else "models/sequence_lap_delta.pt",
                ),
                settings=_replace(load_settings(), base_lap_time_s=base),
            )
        )
        bundle = create_model_artifact_bundle(
            model_path=output_path,
            backend=args.backend,
            training_config=training_config,
            evaluation_report=report,
            replay_evaluation_report=replay_report,
            replay_suite_report=replay_suite,
            artifact_root=args.artifact_root or DEFAULT_ARTIFACT_ROOT,
            replay_dataset=args.replay_dataset,
        )
        print(f"bundled artifact: {bundle.artifact_id}")
        print(f"registry: {bundle.registry_path}")


def _training_config(
    backend: str,
    output_path: Path,
    laps: int,
    seeds: int,
    rounds: int,
    real_data_paths: list[str] | None = None,
    replay_dataset_path: str | None = None,
) -> dict[str, object]:
    training_rows = None
    manifest_path = model_manifest_path(output_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training_rows = manifest.get("training_rows")
    return {
        "backend": backend,
        "output": str(output_path),
        "laps": laps,
        "seeds": seeds,
        "rounds": rounds,
        "training_rows": training_rows,
        "real_data_paths": real_data_paths,
        "replay_dataset_path": replay_dataset_path,
        "feature_schema_hash": feature_schema_hash(),
    }


def _serving_model_for_artifact(backend: str, output_path: Path):
    output = str(output_path)
    return create_serving_model(
        config=ModelConfig(),
        backend=backend,
        xgboost_model_path=output if backend == "xgboost" else "models/xgboost_lap_delta.json",
        lightgbm_model_path=output if backend == "lightgbm" else "models/lightgbm_lap_delta.txt",
        catboost_model_path=output if backend == "catboost" else "models/catboost_lap_delta.cbm",
        sequence_model_path=output if backend == "sequence" else "models/sequence_lap_delta.pt",
    )


if __name__ == "__main__":
    main()
