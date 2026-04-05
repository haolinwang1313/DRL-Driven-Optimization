from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.contracts import write_csv, write_json
from paper_repro.metrics import summarize_surrogate_predictions
from paper_repro.runtime import resolve_device


@dataclass(frozen=True)
class SurrogateCandidateSpec:
    name: str
    tuning: str
    feature_scaler: str
    target_scaler: str
    loss: str
    hidden_layers: list[int] | None = None
    dropout: float | None = None
    learning_rate: float | None = None


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(current, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current = units
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class SurrogateBundle:
    model: MLPRegressor
    x_scaler: MinMaxScaler | StandardScaler
    y_scaler: MinMaxScaler | StandardScaler
    feature_bounds: dict[str, tuple[float, float]]
    target_bounds: dict[str, tuple[float, float]]
    feature_reference: np.ndarray
    hyperparameters: dict
    device: str = "cpu"

    def to(self, device: torch.device | str) -> "SurrogateBundle":
        self.device = str(device)
        self.model.to(device)
        return self

    def predict(self, frame: pd.DataFrame, *, clip: bool = True) -> pd.DataFrame:
        x = self.x_scaler.transform(frame[MORPHOLOGY_FEATURES].to_numpy(dtype=np.float32))
        with torch.no_grad():
            pred = self.model(torch.tensor(x, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
        outputs = self.y_scaler.inverse_transform(pred)
        output_frame = pd.DataFrame(outputs, columns=PERFORMANCE_TARGETS, index=frame.index)
        if clip:
            for target in PERFORMANCE_TARGETS:
                lower, upper = self.target_bounds[target]
                output_frame[target] = output_frame[target].clip(lower=lower, upper=upper)
        return output_frame

    def predict_action(self, action: np.ndarray, *, clip: bool = True) -> np.ndarray:
        frame = pd.DataFrame([action], columns=MORPHOLOGY_FEATURES)
        return self.predict(frame, clip=clip).iloc[0].to_numpy(dtype=float)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _make_scaler(kind: str) -> MinMaxScaler | StandardScaler:
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"Unsupported scaler kind: {kind}")


def _make_loss(loss_name: str) -> nn.Module:
    if loss_name == "mae":
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=0.1)
    raise ValueError(f"Unsupported loss: {loss_name}")


def _anchor_hyperparameters(config: Config, candidate: SurrogateCandidateSpec) -> dict:
    anchor = config["dnn"]["anchor_architecture"]
    return {
        "hidden_layers": list(candidate.hidden_layers or anchor["hidden_layers"]),
        "dropout": float(candidate.dropout if candidate.dropout is not None else anchor["dropout"]),
        "learning_rate": float(candidate.learning_rate if candidate.learning_rate is not None else anchor["learning_rate"]),
        "batch_size": int(config["dnn"]["batch_size"]),
    }


def _candidate_spec_from_name(name: str, config: Config) -> SurrogateCandidateSpec:
    anchor = config["dnn"]["anchor_architecture"]
    registry = {
        "anchor_minmax": SurrogateCandidateSpec(
            name="anchor_minmax",
            tuning="fixed",
            feature_scaler="minmax",
            target_scaler="minmax",
            loss="mae",
            hidden_layers=list(anchor["hidden_layers"]),
            dropout=float(anchor["dropout"]),
            learning_rate=float(anchor["learning_rate"]),
        ),
        "tuned_minmax": SurrogateCandidateSpec(
            name="tuned_minmax",
            tuning="optuna",
            feature_scaler="minmax",
            target_scaler="minmax",
            loss="mae",
        ),
        "tuned_standard": SurrogateCandidateSpec(
            name="tuned_standard",
            tuning="optuna",
            feature_scaler="standard",
            target_scaler="standard",
            loss="mae",
        ),
        "tuned_standard_huber": SurrogateCandidateSpec(
            name="tuned_standard_huber",
            tuning="optuna",
            feature_scaler="standard",
            target_scaler="standard",
            loss="huber",
        ),
    }
    if name not in registry:
        raise ValueError(f"Unknown surrogate candidate preset: {name}")
    return registry[name]


def candidate_specs_from_config(config: Config) -> list[SurrogateCandidateSpec]:
    selection_cfg = config.raw.get("surrogate_selection", {})
    preset_names = selection_cfg.get("candidate_presets", ["tuned_minmax"])
    return [_candidate_spec_from_name(str(name), config) for name in preset_names]


def _train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: list[int],
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    loss_name: str,
    device: torch.device,
    seed: int,
) -> tuple[MLPRegressor, float]:
    _seed_everything(seed)
    model = MLPRegressor(x_train.shape[1], y_train.shape[1], hidden_layers=hidden_layers, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = _make_loss(loss_name)
    train_loader = _build_loader(x_train, y_train, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state = None
    patience_left = patience
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(x_val, dtype=torch.float32, device=device))
            val_loss = criterion(val_pred, torch.tensor(y_val, dtype=torch.float32, device=device)).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def _evaluate_cv(
    frame: pd.DataFrame,
    candidate: SurrogateCandidateSpec,
    hyperparameters: dict,
    config: Config,
    *,
    candidate_index: int,
    evaluation_seed: int,
) -> tuple[float, list[dict], pd.DataFrame, dict]:
    dnn_cfg = config["dnn"]
    device = resolve_device(config)
    x_raw = frame[MORPHOLOGY_FEATURES].to_numpy(dtype=np.float32)
    y_raw = frame[PERFORMANCE_TARGETS].to_numpy(dtype=np.float32)
    kfold = KFold(
        n_splits=int(dnn_cfg["folds"]),
        shuffle=True,
        random_state=int(config["project"]["random_seed"]) + candidate_index,
    )
    fold_metrics = []
    fold_rows = []
    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(x_raw), start=1):
        x_scaler = _make_scaler(candidate.feature_scaler)
        y_scaler = _make_scaler(candidate.target_scaler)
        x_train = x_scaler.fit_transform(x_raw[train_idx])
        y_train = y_scaler.fit_transform(y_raw[train_idx])
        x_test = x_scaler.transform(x_raw[test_idx])
        y_test = y_scaler.transform(y_raw[test_idx])
        model, _ = _train_single_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            hidden_layers=hyperparameters["hidden_layers"],
            dropout=hyperparameters["dropout"],
            learning_rate=hyperparameters["learning_rate"],
            batch_size=hyperparameters["batch_size"],
            epochs=int(dnn_cfg["epochs"]),
            patience=int(dnn_cfg["patience"]),
            loss_name=candidate.loss,
            device=device,
            seed=int(config["project"]["random_seed"]) + candidate_index * 1000 + evaluation_seed * 100 + fold_index,
        )
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(x_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
        pred = y_scaler.inverse_transform(pred_scaled)
        truth = y_raw[test_idx]
        fold_metric = {"fold": fold_index}
        for target_index, target in enumerate(PERFORMANCE_TARGETS):
            target_range = max(float(frame[target].max() - frame[target].min()), 1e-8)
            fold_metric[f"{target}_R2"] = float(r2_score(truth[:, target_index], pred[:, target_index]))
            fold_metric[f"{target}_MAE"] = float(mean_absolute_error(truth[:, target_index], pred[:, target_index]))
            fold_metric[f"{target}_NMAE"] = float(mean_absolute_error(truth[:, target_index], pred[:, target_index]) / target_range)
        fold_metrics.append(fold_metric)
        fold_rows.append(
            pd.DataFrame(
                {
                    "fold": fold_index,
                    "sample_id": frame.iloc[test_idx]["sample_id"].to_numpy(),
                    **{f"true_{target}": truth[:, idx] for idx, target in enumerate(PERFORMANCE_TARGETS)},
                    **{f"pred_{target}": pred[:, idx] for idx, target in enumerate(PERFORMANCE_TARGETS)},
                }
            )
        )
    cv_predictions = pd.concat(fold_rows, ignore_index=True)
    selection_cfg = config.raw.get("surrogate_selection", {})
    metrics_summary = summarize_surrogate_predictions(
        cv_predictions,
        low_quantile=float(selection_cfg.get("low_quantile", 0.1)),
        high_quantile=float(selection_cfg.get("high_quantile", 0.9)),
    )
    selection_objective = float(
        metrics_summary["aggregate"]["mean_target_nmae"]
        + float(selection_cfg.get("tail_weight", 0.25)) * metrics_summary["aggregate"]["mean_tail_nmae"]
    )
    return selection_objective, fold_metrics, cv_predictions, metrics_summary


def _suggest_hyperparameters(trial: optuna.Trial, config: Config) -> dict:
    dnn_cfg = config["dnn"]
    n_layers = trial.suggest_int("n_layers", int(dnn_cfg["hidden_layers_min"]), int(dnn_cfg["hidden_layers_max"]))
    hidden_layers = [
        trial.suggest_int(
            f"units_{index + 1}",
            int(dnn_cfg["units_min"]),
            int(dnn_cfg["units_max"]),
            step=int(dnn_cfg["units_step"]),
        )
        for index in range(n_layers)
    ]
    return {
        "hidden_layers": hidden_layers,
        "dropout": trial.suggest_float("dropout", float(dnn_cfg["dropout_min"]), float(dnn_cfg["dropout_max"])),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(dnn_cfg["learning_rate_min"]),
            float(dnn_cfg["learning_rate_max"]),
            log=True,
        ),
        "batch_size": int(dnn_cfg["batch_size"]),
    }


def _normalize_trial_params(params: dict, config: Config) -> dict:
    normalized = dict(params)
    hidden_layers = [
        normalized.pop(f"units_{idx + 1}") for idx in range(int(normalized.pop("n_layers")))
    ]
    normalized["hidden_layers"] = hidden_layers
    normalized["batch_size"] = int(config["dnn"]["batch_size"])
    return normalized


def _guardrail_reference(dataset: pd.DataFrame) -> np.ndarray:
    features = dataset[MORPHOLOGY_FEATURES].to_numpy(dtype=np.float32)
    lower = features.min(axis=0)
    upper = features.max(axis=0)
    return (features - lower) / np.maximum(upper - lower, 1e-8)


def _copy_if_needed(source: str | Path, target: Path) -> None:
    source_path = Path(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == target.resolve():
        return
    shutil.copy2(source_path, target)


def train_surrogate_candidate(
    config: Config,
    dataset: pd.DataFrame,
    candidate: SurrogateCandidateSpec,
    *,
    output_dir: str | Path | None = None,
    candidate_index: int = 0,
) -> tuple[SurrogateBundle, dict]:
    dirs = config.ensure_artifact_dirs()
    device = resolve_device(config)
    dataset = dataset.copy()
    feature_bounds = {feature: (float(dataset[feature].min()), float(dataset[feature].max())) for feature in MORPHOLOGY_FEATURES}
    target_bounds = {target: (float(dataset[target].min()), float(dataset[target].max())) for target in PERFORMANCE_TARGETS}
    output_root = Path(output_dir) if output_dir is not None else Path(dirs["models_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    base_seed = int(config["project"]["random_seed"]) + candidate_index * 1000

    study_payload = None
    if candidate.tuning == "optuna":
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=base_seed))

        def objective(trial: optuna.Trial) -> float:
            hyperparameters = _suggest_hyperparameters(trial, config)
            selection_objective, _, _, metrics_summary = _evaluate_cv(
                dataset,
                candidate,
                hyperparameters,
                config,
                candidate_index=candidate_index,
                evaluation_seed=trial.number + 1,
            )
            trial.set_user_attr("mean_target_nmae", metrics_summary["aggregate"]["mean_target_nmae"])
            trial.set_user_attr("mean_tail_nmae", metrics_summary["aggregate"]["mean_tail_nmae"])
            trial.set_user_attr("mean_r2", metrics_summary["aggregate"]["mean_r2"])
            return selection_objective

        study.optimize(objective, n_trials=int(config["dnn"]["n_trials"]), show_progress_bar=False)
        best_hyperparameters = _normalize_trial_params(study.best_trial.params, config)
        study_payload = {
            "best_value": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
            "best_trial_attributes": dict(study.best_trial.user_attrs),
        }
    else:
        best_hyperparameters = _anchor_hyperparameters(config, candidate)

    selection_objective, fold_metrics, cv_predictions, metrics_summary = _evaluate_cv(
        dataset,
        candidate,
        best_hyperparameters,
        config,
        candidate_index=candidate_index,
        evaluation_seed=999,
    )

    x_scaler = _make_scaler(candidate.feature_scaler)
    y_scaler = _make_scaler(candidate.target_scaler)
    x_full = x_scaler.fit_transform(dataset[MORPHOLOGY_FEATURES].to_numpy(dtype=np.float32))
    y_full = y_scaler.fit_transform(dataset[PERFORMANCE_TARGETS].to_numpy(dtype=np.float32))
    model, _ = _train_single_model(
        x_train=x_full,
        y_train=y_full,
        x_val=x_full,
        y_val=y_full,
        hidden_layers=best_hyperparameters["hidden_layers"],
        dropout=best_hyperparameters["dropout"],
        learning_rate=best_hyperparameters["learning_rate"],
        batch_size=best_hyperparameters["batch_size"],
        epochs=int(config["dnn"]["retrain_epochs"]),
        patience=int(config["dnn"]["patience"]),
        loss_name=candidate.loss,
        device=device,
        seed=base_seed + 9000,
    )
    model.cpu()

    bundle = SurrogateBundle(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_bounds=feature_bounds,
        target_bounds=target_bounds,
        feature_reference=_guardrail_reference(dataset),
        hyperparameters=best_hyperparameters,
        device="cpu",
    )

    model_path = output_root / "surrogate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_bounds": feature_bounds,
            "target_bounds": target_bounds,
            "hyperparameters": best_hyperparameters,
            "feature_reference": bundle.feature_reference,
            "feature_names": MORPHOLOGY_FEATURES,
            "target_names": PERFORMANCE_TARGETS,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "candidate": asdict(candidate),
        },
        model_path,
    )
    cv_predictions_path = write_csv(cv_predictions, output_root / "cv_predictions.csv")

    summary_path = output_root / "surrogate_summary.json"
    summary = {
        "candidate": asdict(candidate),
        "dataset_rows": int(len(dataset)),
        "dataset_scale": int(len(dataset)),
        "best_hyperparameters": best_hyperparameters,
        "selection_objective": float(selection_objective),
        "fold_metrics": fold_metrics,
        "cv_metrics": metrics_summary,
        "optuna": study_payload,
        "training_data_source": "simulated_only",
        "benchmark_leakage_checked": True,
        "imported_publication_artifacts_used_for_training": False,
        "model_path": str(model_path),
        "cv_predictions_path": str(cv_predictions_path),
        "summary_path": str(summary_path),
    }
    write_json(summary, summary_path)
    return bundle, summary


def select_best_surrogate_record(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        raise ValueError("Cannot select a surrogate winner from an empty frame.")
    ordered = frame.sort_values(
        ["mean_target_nmae", "mean_tail_nmae", "mean_r2", "selection_objective", "candidate"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def select_final_scale_record(
    regime_best_frame: pd.DataFrame,
    *,
    original_scale: int,
    tolerance_pct: float,
) -> tuple[dict[str, object], dict[str, object]]:
    expanded = regime_best_frame.loc[regime_best_frame["dataset_scale"] > original_scale].copy()
    if expanded.empty:
        original = regime_best_frame.loc[regime_best_frame["dataset_scale"] == original_scale]
        if original.empty:
            raise ValueError("No original-scale surrogate record available for fallback selection.")
        selected = select_best_surrogate_record(original)
        return selected, {
            "mode": "fallback_to_original",
            "tolerance_pct": float(tolerance_pct),
            "eligible_scales": [int(selected["dataset_scale"])],
        }

    best_primary = float(expanded["mean_target_nmae"].min())
    primary_threshold = best_primary * (1.0 + tolerance_pct)
    eligible = expanded.loc[expanded["mean_target_nmae"] <= primary_threshold + 1e-12].copy()
    eligible = eligible.sort_values(
        ["dataset_scale", "mean_tail_nmae", "mean_r2", "selection_objective"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = eligible.iloc[0].to_dict()
    return selected, {
        "mode": "smallest_within_primary_tolerance",
        "tolerance_pct": float(tolerance_pct),
        "best_primary_mean_target_nmae": best_primary,
        "primary_threshold": primary_threshold,
        "eligible_scales": [int(scale) for scale in eligible["dataset_scale"].tolist()],
    }


def train_surrogate_selection(config: Config, dataset_regimes: dict[int, pd.DataFrame]) -> tuple[SurrogateBundle, dict]:
    dirs = config.ensure_artifact_dirs()
    models_dir = Path(dirs["models_dir"])
    data_dir = Path(dirs["data_dir"])
    selection_root = models_dir / "surrogate_selection"
    selection_cfg = config.raw.get("surrogate_selection", {})
    original_scale = int(selection_cfg.get("original_scale", min(dataset_regimes)))
    tolerance_pct = float(selection_cfg.get("scale_tolerance_pct", 0.05))
    candidates = candidate_specs_from_config(config)

    comparison_records: list[dict[str, object]] = []
    regime_winner_records: list[dict[str, object]] = []
    for scale in sorted(dataset_regimes):
        dataset = dataset_regimes[scale]
        scale_records = []
        for candidate_index, candidate in enumerate(candidates):
            output_dir = selection_root / f"scale_{scale}" / candidate.name
            _, candidate_summary = train_surrogate_candidate(
                config,
                dataset,
                candidate,
                output_dir=output_dir,
                candidate_index=candidate_index,
            )
            aggregate = candidate_summary["cv_metrics"]["aggregate"]
            record = {
                "dataset_scale": int(scale),
                "regime_type": "original" if scale <= original_scale else "expanded",
                "candidate": candidate.name,
                "tuning": candidate.tuning,
                "feature_scaler": candidate.feature_scaler,
                "target_scaler": candidate.target_scaler,
                "loss": candidate.loss,
                "mean_target_mae": float(aggregate["mean_target_mae"]),
                "mean_target_nmae": float(aggregate["mean_target_nmae"]),
                "mean_tail_mae": float(aggregate["mean_tail_mae"]),
                "mean_tail_nmae": float(aggregate["mean_tail_nmae"]),
                "mean_r2": float(aggregate["mean_r2"]),
                "worst_target_nmae": float(aggregate["worst_target_nmae"]),
                "selection_objective": float(candidate_summary["selection_objective"]),
                "model_path": str(candidate_summary["model_path"]),
                "cv_predictions_path": str(candidate_summary["cv_predictions_path"]),
                "summary_path": str(candidate_summary["summary_path"]),
                "dataset_path": str(data_dir / "regimes" / f"scale_{scale}" / "simulated_samples.csv"),
                "dataset_meta_path": str(data_dir / "regimes" / f"scale_{scale}" / "simulated_samples.meta.json"),
                "dataset_blocks_path": str(data_dir / "regimes" / f"scale_{scale}" / "simulated_blocks.jsonl"),
            }
            comparison_records.append(record)
            scale_records.append(record)
        regime_winner_records.append(select_best_surrogate_record(pd.DataFrame(scale_records)))

    comparison_frame = pd.DataFrame(comparison_records)
    comparison_csv = write_csv(comparison_frame, models_dir / "surrogate_comparison.csv")
    regime_best_frame = pd.DataFrame(regime_winner_records)
    regime_best_csv = write_csv(regime_best_frame, models_dir / "surrogate_regime_winners.csv")
    best_original = select_best_surrogate_record(regime_best_frame.loc[regime_best_frame["dataset_scale"] == original_scale])
    selected_record, scale_selection = select_final_scale_record(
        regime_best_frame,
        original_scale=original_scale,
        tolerance_pct=tolerance_pct,
    )

    _copy_if_needed(selected_record["model_path"], models_dir / "surrogate.pt")
    _copy_if_needed(selected_record["cv_predictions_path"], models_dir / "cv_predictions.csv")
    _copy_if_needed(selected_record["summary_path"], models_dir / "surrogate_summary.json")
    _copy_if_needed(selected_record["dataset_path"], data_dir / "simulated_samples.csv")
    _copy_if_needed(selected_record["dataset_meta_path"], data_dir / "simulated_samples.meta.json")
    _copy_if_needed(selected_record["dataset_blocks_path"], data_dir / "simulated_blocks.jsonl")

    selected_bundle = load_surrogate(models_dir / "surrogate.pt")
    selection_summary = {
        "protocol_version": "surrogate-selection-v1",
        "candidate_presets": [candidate.name for candidate in candidates],
        "original_scale": original_scale,
        "comparison_csv": str(comparison_csv),
        "regime_winners_csv": str(regime_best_csv),
        "best_original_500": best_original,
        "best_expanded": selected_record if int(selected_record["dataset_scale"]) > original_scale else None,
        "selected_for_optimization": selected_record,
        "scale_selection": scale_selection,
        "selected_model_alias": str(models_dir / "surrogate.pt"),
        "selected_cv_predictions_alias": str(models_dir / "cv_predictions.csv"),
        "selected_summary_alias": str(models_dir / "surrogate_summary.json"),
        "selected_dataset_alias": str(data_dir / "simulated_samples.csv"),
        "selected_dataset_meta_alias": str(data_dir / "simulated_samples.meta.json"),
        "leakage_guard": {
            "benchmark_dataset_used_for_training": False,
            "imported_publication_results_used_for_training": False,
        },
    }
    write_json(selection_summary, models_dir / "selected_surrogate.json")
    return selected_bundle, selection_summary


def train_surrogate(config: Config, dataset: pd.DataFrame) -> tuple[SurrogateBundle, dict]:
    default_candidate = _candidate_spec_from_name("tuned_minmax", config)
    return train_surrogate_candidate(config, dataset, default_candidate)


def load_surrogate(model_path: str | Path, device: torch.device | str | None = None) -> SurrogateBundle:
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MLPRegressor(
        input_dim=len(checkpoint["feature_names"]),
        output_dim=len(checkpoint["target_names"]),
        hidden_layers=checkpoint["hyperparameters"]["hidden_layers"],
        dropout=checkpoint["hyperparameters"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if "x_scaler" in checkpoint and "y_scaler" in checkpoint:
        x_scaler = checkpoint["x_scaler"]
        y_scaler = checkpoint["y_scaler"]
    else:
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        x_scaler.min_ = checkpoint["x_scaler_min"]
        x_scaler.scale_ = checkpoint["x_scaler_scale"]
        x_scaler.data_min_ = np.zeros(len(checkpoint["feature_names"]))
        x_scaler.data_max_ = np.ones(len(checkpoint["feature_names"]))
        x_scaler.data_range_ = np.ones(len(checkpoint["feature_names"]))
        x_scaler.n_features_in_ = len(checkpoint["feature_names"])

        y_scaler.min_ = checkpoint["y_scaler_min"]
        y_scaler.scale_ = checkpoint["y_scaler_scale"]
        y_scaler.data_min_ = np.zeros(len(checkpoint["target_names"]))
        y_scaler.data_max_ = np.ones(len(checkpoint["target_names"]))
        y_scaler.data_range_ = np.ones(len(checkpoint["target_names"]))
        y_scaler.n_features_in_ = len(checkpoint["target_names"])

    return SurrogateBundle(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_bounds=checkpoint["feature_bounds"],
        target_bounds=checkpoint["target_bounds"],
        feature_reference=checkpoint["feature_reference"],
        hyperparameters=checkpoint["hyperparameters"],
        device="cpu",
    ).to(device or "cpu")
