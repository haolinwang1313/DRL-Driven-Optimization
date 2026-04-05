from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.contracts import write_csv, write_json
from paper_repro.runtime import resolve_device


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
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
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


def _build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    device: torch.device,
) -> tuple[MLPRegressor, float]:
    model = MLPRegressor(x_train.shape[1], y_train.shape[1], hidden_layers=hidden_layers, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
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


def _evaluate_cv(frame: pd.DataFrame, hyperparameters: dict, config: Config) -> tuple[float, list[dict], pd.DataFrame]:
    dnn_cfg = config["dnn"]
    device = resolve_device(config)
    X = frame[MORPHOLOGY_FEATURES].to_numpy(dtype=np.float32)
    y = frame[PERFORMANCE_TARGETS].to_numpy(dtype=np.float32)
    kfold = KFold(n_splits=dnn_cfg["folds"], shuffle=True, random_state=config["project"]["random_seed"])
    fold_metrics = []
    fold_rows = []
    losses = []
    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        x_train = x_scaler.fit_transform(X[train_idx])
        y_train = y_scaler.fit_transform(y[train_idx])
        x_test = x_scaler.transform(X[test_idx])
        y_test = y_scaler.transform(y[test_idx])
        model, _ = _train_single_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_test,
            y_val=y_test,
            hidden_layers=hyperparameters["hidden_layers"],
            dropout=hyperparameters["dropout"],
            learning_rate=hyperparameters["learning_rate"],
            batch_size=hyperparameters["batch_size"],
            epochs=dnn_cfg["epochs"],
            patience=dnn_cfg["patience"],
            device=device,
        )
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(x_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
        loss = mean_absolute_error(y_test, pred_scaled)
        losses.append(loss)
        pred = y_scaler.inverse_transform(pred_scaled)
        truth = y[test_idx]
        fold_metric = {"fold": fold_index}
        for target_index, target in enumerate(PERFORMANCE_TARGETS):
            fold_metric[f"{target}_R2"] = float(r2_score(truth[:, target_index], pred[:, target_index]))
            fold_metric[f"{target}_MAE"] = float(mean_absolute_error(y_test[:, target_index], pred_scaled[:, target_index]))
        fold_metrics.append(fold_metric)
        fold_frame = pd.DataFrame(
            {
                "fold": fold_index,
                "sample_id": frame.iloc[test_idx]["sample_id"].to_numpy(),
                **{f"true_{target}": truth[:, idx] for idx, target in enumerate(PERFORMANCE_TARGETS)},
                **{f"pred_{target}": pred[:, idx] for idx, target in enumerate(PERFORMANCE_TARGETS)},
            }
        )
        fold_rows.append(fold_frame)
    return float(np.mean(losses)), fold_metrics, pd.concat(fold_rows, ignore_index=True)


def _suggest_hyperparameters(trial: optuna.Trial, config: Config) -> dict:
    dnn_cfg = config["dnn"]
    n_layers = trial.suggest_int("n_layers", dnn_cfg["hidden_layers_min"], dnn_cfg["hidden_layers_max"])
    hidden_layers = [
        trial.suggest_int(
            f"units_{index + 1}",
            dnn_cfg["units_min"],
            dnn_cfg["units_max"],
            step=dnn_cfg["units_step"],
        )
        for index in range(n_layers)
    ]
    return {
        "hidden_layers": hidden_layers,
        "dropout": trial.suggest_float("dropout", dnn_cfg["dropout_min"], dnn_cfg["dropout_max"]),
        "learning_rate": trial.suggest_float("learning_rate", dnn_cfg["learning_rate_min"], dnn_cfg["learning_rate_max"], log=True),
        "batch_size": dnn_cfg["batch_size"],
    }


def train_surrogate(config: Config, dataset: pd.DataFrame) -> tuple[SurrogateBundle, dict]:
    dirs = config.ensure_artifact_dirs()
    device = resolve_device(config)
    dataset = dataset.copy()
    feature_bounds = {feature: (float(dataset[feature].min()), float(dataset[feature].max())) for feature in MORPHOLOGY_FEATURES}
    target_bounds = {target: (float(dataset[target].min()), float(dataset[target].max())) for target in PERFORMANCE_TARGETS}

    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.Trial) -> float:
        hyperparameters = _suggest_hyperparameters(trial, config)
        mean_loss, _, _ = _evaluate_cv(dataset, hyperparameters, config)
        return mean_loss

    study.optimize(objective, n_trials=config["dnn"]["n_trials"], show_progress_bar=False)
    best_hyperparameters = study.best_trial.params
    best_hyperparameters["hidden_layers"] = [
        best_hyperparameters.pop(f"units_{idx + 1}") for idx in range(best_hyperparameters.pop("n_layers"))
    ]
    best_hyperparameters["batch_size"] = config["dnn"]["batch_size"]

    mean_loss, fold_metrics, cv_predictions = _evaluate_cv(dataset, best_hyperparameters, config)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(dataset[MORPHOLOGY_FEATURES].to_numpy(dtype=np.float32))
    y = y_scaler.fit_transform(dataset[PERFORMANCE_TARGETS].to_numpy(dtype=np.float32))
    model, _ = _train_single_model(
        x_train=X,
        y_train=y,
        x_val=X,
        y_val=y,
        hidden_layers=best_hyperparameters["hidden_layers"],
        dropout=best_hyperparameters["dropout"],
        learning_rate=best_hyperparameters["learning_rate"],
        batch_size=best_hyperparameters["batch_size"],
        epochs=config["dnn"]["retrain_epochs"],
        patience=config["dnn"]["patience"],
        device=device,
    )
    model.cpu()

    bundle = SurrogateBundle(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_bounds=feature_bounds,
        target_bounds=target_bounds,
        feature_reference=X.copy(),
        hyperparameters=best_hyperparameters,
        device="cpu",
    )

    model_path = Path(dirs["models_dir"]) / "surrogate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_bounds": feature_bounds,
            "target_bounds": target_bounds,
            "hyperparameters": best_hyperparameters,
            "feature_reference": X,
            "x_scaler_min": x_scaler.min_,
            "x_scaler_scale": x_scaler.scale_,
            "y_scaler_min": y_scaler.min_,
            "y_scaler_scale": y_scaler.scale_,
            "feature_names": MORPHOLOGY_FEATURES,
            "target_names": PERFORMANCE_TARGETS,
        },
        model_path,
    )
    write_csv(cv_predictions, Path(dirs["models_dir"]) / "cv_predictions.csv")

    summary = {
        "best_hyperparameters": best_hyperparameters,
        "mean_cv_mae": mean_loss,
        "fold_metrics": fold_metrics,
        "optuna_best_value": study.best_value,
        "anchor_architecture": config["dnn"]["anchor_architecture"],
    }
    write_json(summary, Path(dirs["models_dir"]) / "surrogate_summary.json")
    return bundle, summary


def load_surrogate(model_path: str | Path, device: torch.device | str | None = None) -> SurrogateBundle:
    # Older checkpoints may reference numpy._core during unpickling.
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
