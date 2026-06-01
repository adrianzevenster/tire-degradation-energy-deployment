from __future__ import annotations

import argparse
import json
from pathlib import Path

from f1_strategy.artifacts import DEFAULT_ARTIFACT_ROOT, create_model_artifact_bundle
from f1_strategy.config import load_settings
from f1_strategy.domain import TireCompound
from f1_strategy.evaluation import run_evaluation
from f1_strategy.feature_store import OnlineFeatureStore
from f1_strategy.models import (
    FEATURE_NAMES,
    feature_schema_hash,
    features_to_vector,
    model_manifest_path,
    write_model_manifest,
)
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
    return parser


def _training_rows(laps: int, seeds: int) -> tuple[list[list[float]], list[float]]:
    settings = load_settings()
    rows: list[list[float]] = []
    targets: list[float] = []
    compounds = list(TireCompound)

    for seed in range(1, seeds + 1):
        compound = compounds[(seed - 1) % len(compounds)]
        simulator = RaceSimulator(
            SimulationConfig(
                session_id=f"train-{seed}",
                car_id=f"car-{seed % 20:02d}",
                laps=laps,
                seed=seed,
                compound=compound,
                base_lap_time_s=settings.base_lap_time_s,
            )
        )
        store = OnlineFeatureStore(window_size=settings.feature_window_size)
        for event in simulator.events():
            features = store.ingest(event)
            if event.lap_time_s is None:
                continue
            rows.append(features_to_vector(features))
            targets.append(event.lap_time_s - settings.base_lap_time_s)

    return rows, targets


def _default_output(backend: str) -> str:
    return {
        "xgboost": "models/xgboost_lap_delta.json",
        "lightgbm": "models/lightgbm_lap_delta.txt",
        "catboost": "models/catboost_lap_delta.cbm",
        "sequence": "models/sequence_lap_delta.pt",
    }[backend]


def train_xgboost_model(output: str, laps: int, seeds: int, rounds: int) -> Path:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError('Install ML dependencies first: pip install -e ".[ml]"') from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds)
    matrix = xgb.DMatrix(rows, label=targets, feature_names=FEATURE_NAMES)
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
    booster = xgb.train(params=params, dtrain=matrix, num_boost_round=rounds)
    booster.set_attr(
        model_type="lap_delta_regressor",
        feature_schema=",".join(FEATURE_NAMES),
        feature_schema_hash=feature_schema_hash(),
        training_rows=str(len(rows)),
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    write_model_manifest(output_path, backend="xgboost", training_rows=len(rows))
    return output_path


def train_lightgbm_model(output: str, laps: int, seeds: int, rounds: int) -> Path:
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Install LightGBM dependencies first: pip install -e ".[ml]"') from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds)
    dataset = lgb.Dataset(
        np.asarray(rows, dtype="float32"),
        label=np.asarray(targets, dtype="float32"),
        feature_name=FEATURE_NAMES,
    )
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
    booster = lgb.train(params=params, train_set=dataset, num_boost_round=rounds)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    write_model_manifest(output_path, backend="lightgbm", training_rows=len(rows))
    return output_path


def train_catboost_model(output: str, laps: int, seeds: int, rounds: int) -> Path:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise RuntimeError("Install CatBoost first: pip install -e '.[catboost]'") from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds)
    model = CatBoostRegressor(
        iterations=rounds,
        depth=5,
        learning_rate=0.055,
        loss_function="RMSE",
        random_seed=7,
        verbose=False,
    )
    model.fit(Pool(rows, targets, feature_names=FEATURE_NAMES))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    write_model_manifest(output_path, backend="catboost", training_rows=len(rows))
    return output_path


def train_sequence_model(output: str, laps: int, seeds: int, rounds: int) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install Torch first: pip install -e '.[deep]'") from exc

    rows, targets = _training_rows(laps=laps, seeds=seeds)
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
    return output_path


def train_model(backend: str, output: str, laps: int, seeds: int, rounds: int) -> Path:
    if backend == "xgboost":
        return train_xgboost_model(output, laps, seeds, rounds)
    if backend == "lightgbm":
        return train_lightgbm_model(output, laps, seeds, rounds)
    if backend == "catboost":
        return train_catboost_model(output, laps, seeds, rounds)
    if backend == "sequence":
        return train_sequence_model(output, laps, seeds, rounds)
    raise ValueError(f"Unsupported training backend: {backend}")


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or _default_output(args.backend)
    output_path = train_model(
        backend=args.backend,
        output=output,
        laps=args.laps,
        seeds=args.seeds,
        rounds=args.rounds,
    )
    print(f"saved {args.backend} model: {output_path}")
    if args.artifact_root is not None:
        training_config = _training_config(
            backend=args.backend,
            output_path=output_path,
            laps=args.laps,
            seeds=args.seeds,
            rounds=args.rounds,
        )
        report = run_evaluation(
            model_backend=args.backend,
            model_paths={args.backend: str(output_path), "sequence": str(output_path)},
        )
        bundle = create_model_artifact_bundle(
            model_path=output_path,
            backend=args.backend,
            training_config=training_config,
            evaluation_report=report,
            artifact_root=args.artifact_root or DEFAULT_ARTIFACT_ROOT,
        )
        print(f"bundled artifact: {bundle.artifact_id}")
        print(f"registry: {bundle.registry_path}")


def _training_config(
    backend: str,
    output_path: Path,
    laps: int,
    seeds: int,
    rounds: int,
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
        "feature_schema_hash": feature_schema_hash(),
    }


if __name__ == "__main__":
    main()
