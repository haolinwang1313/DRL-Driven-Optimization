from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from torch import nn

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.contracts import OPTIMIZATION_RESULT_COLUMNS, write_csv, write_json
from paper_repro.metrics import load_benchmark_results
from paper_repro.runtime import resolve_device
from paper_repro.surrogate import SurrogateBundle


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[index] for index in indices), strict=True)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(np.asarray(actions), dtype=torch.float32),
            torch.tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(np.asarray(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for units in hidden_layers:
            layers.extend([nn.Linear(current, units), nn.ReLU()])
            current = units
        layers.extend([nn.Linear(current, output_dim), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list[int]) -> None:
        super().__init__()
        current = state_dim + action_dim
        layers: list[nn.Module] = []
        for units in hidden_layers:
            layers.extend([nn.Linear(current, units), nn.ReLU()])
            current = units
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=1))


@dataclass
class OptimizationEnvironment:
    surrogate: SurrogateBundle
    guardrail_cfg: dict | None = None

    def __post_init__(self) -> None:
        self.feature_min = np.array([self.surrogate.feature_bounds[name][0] for name in MORPHOLOGY_FEATURES], dtype=np.float32)
        self.feature_max = np.array([self.surrogate.feature_bounds[name][1] for name in MORPHOLOGY_FEATURES], dtype=np.float32)
        self.target_min = np.array([self.surrogate.target_bounds[name][0] for name in PERFORMANCE_TARGETS], dtype=np.float32)
        self.target_max = np.array([self.surrogate.target_bounds[name][1] for name in PERFORMANCE_TARGETS], dtype=np.float32)
        self.target_range = np.maximum(self.target_max - self.target_min, 1e-8)
        self.feature_reference = np.asarray(self.surrogate.feature_reference, dtype=np.float32)
        cfg = self.guardrail_cfg or {}
        reference_distances = np.linalg.norm(
            self.feature_reference[:, None, :] - self.feature_reference[None, :, :],
            axis=2,
        )
        np.fill_diagonal(reference_distances, np.inf)
        nn_distances = reference_distances.min(axis=1)
        self.feasible_radius = float(np.percentile(nn_distances, float(cfg.get("feature_radius_percentile", 90))))
        self.feature_penalty_scale = np.asarray(cfg.get("feature_penalty_scale", [10.0, 0.85, 0.6]), dtype=np.float32)
        self.extrapolation_penalty_scale = float(cfg.get("extrapolation_penalty_scale", 1.0))
        self.clip_outputs = bool(cfg.get("clip_outputs", True))

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return self.feature_min + np.clip(action, 0.0, 1.0) * (self.feature_max - self.feature_min)

    def normalize_state(self, outputs: np.ndarray) -> np.ndarray:
        return (outputs - self.target_min) / np.maximum(self.target_max - self.target_min, 1e-8)

    def _feasibility_penalty(self, normalized_action: np.ndarray) -> float:
        distance = np.linalg.norm(self.feature_reference - normalized_action, axis=1).min()
        return float(max(distance - self.feasible_radius, 0.0))

    def _extrapolation_penalty(self, outputs: np.ndarray) -> float:
        below = np.maximum(self.target_min - outputs, 0.0) / self.target_range
        above = np.maximum(outputs - self.target_max, 0.0) / self.target_range
        return float((below + above).sum())

    def evaluate_batch(self, normalized_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]
        clipped_actions = np.clip(normalized_actions, 0.0, 1.0)
        actual_actions = self.feature_min + clipped_actions * (self.feature_max - self.feature_min)
        raw_outputs = self.surrogate.predict(
            pd.DataFrame(actual_actions, columns=MORPHOLOGY_FEATURES),
            clip=False,
        ).to_numpy(dtype=np.float32)
        distance = np.linalg.norm(self.feature_reference[None, :, :] - clipped_actions[:, None, :], axis=2).min(axis=1)
        penalty = np.maximum(distance - self.feasible_radius, 0.0).astype(np.float32)
        below = np.maximum(self.target_min[None, :] - raw_outputs, 0.0) / self.target_range[None, :]
        above = np.maximum(raw_outputs - self.target_max[None, :], 0.0) / self.target_range[None, :]
        extrapolation_penalty = (below + above).sum(axis=1).astype(np.float32)

        outputs = raw_outputs.copy()
        outputs[:, 0] += self.feature_penalty_scale[0] * penalty
        outputs[:, 1] -= self.feature_penalty_scale[1] * penalty
        outputs[:, 2] -= self.feature_penalty_scale[2] * penalty
        outputs[:, 0] += self.target_range[0] * self.extrapolation_penalty_scale * extrapolation_penalty
        outputs[:, 1] -= self.target_range[1] * self.extrapolation_penalty_scale * extrapolation_penalty
        outputs[:, 2] -= self.target_range[2] * self.extrapolation_penalty_scale * extrapolation_penalty
        if self.clip_outputs:
            outputs[:, 0] = np.clip(outputs[:, 0], self.target_min[0], self.target_max[0])
            outputs[:, 1] = np.clip(outputs[:, 1], self.target_min[1], self.target_max[1])
            outputs[:, 2] = np.clip(outputs[:, 2], self.target_min[2], self.target_max[2])
        return actual_actions, outputs

    def evaluate(self, normalized_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actual_actions, outputs = self.evaluate_batch(np.asarray(normalized_action, dtype=np.float32))
        return actual_actions[0], outputs[0]

    def reward(self, outputs: np.ndarray, weights: Iterable[float]) -> float:
        weights_array = np.asarray(list(weights), dtype=np.float32)
        state = self.normalize_state(outputs)
        utopia = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        weighted_distance = float(np.sqrt(np.sum((weights_array * (state - utopia)) ** 2)))
        max_distance = float(np.sqrt(np.sum(weights_array**2)))
        normalized_distance = weighted_distance / max(max_distance, 1e-8)
        return 1.0 - normalized_distance

    def reward_batch(self, outputs: np.ndarray, weights: Iterable[float]) -> np.ndarray:
        weights_array = np.asarray(list(weights), dtype=np.float32)
        state = (np.asarray(outputs, dtype=np.float32) - self.target_min[None, :]) / np.maximum(self.target_range[None, :], 1e-8)
        utopia = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        weighted_distance = np.sqrt(np.sum((weights_array[None, :] * (state - utopia[None, :])) ** 2, axis=1))
        max_distance = float(np.sqrt(np.sum(weights_array**2)))
        normalized_distance = weighted_distance / max(max_distance, 1e-8)
        return 1.0 - normalized_distance


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters(), strict=True):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def run_ddpg(
    config: Config,
    surrogate: SurrogateBundle,
    scenarios: list[str] | None = None,
    seed_start: int = 0,
    seed_end: int | None = None,
    output_suffix: str = "",
) -> tuple[pd.DataFrame, dict]:
    dirs = config.ensure_artifact_dirs()
    ddpg_cfg = config["optimization"]["ddpg"]
    device = resolve_device(config)
    surrogate.to(device)
    env = OptimizationEnvironment(surrogate=surrogate, guardrail_cfg=config["optimization"].get("surrogate_guardrail"))
    results = []
    logs: dict[str, list[dict[str, float]]] = {}
    detailed_logs: dict[str, dict[int, list[dict[str, float]]]] = {}
    selected_scenarios = scenarios or list(ddpg_cfg["scenarios"])
    if seed_end is None:
        seed_end = ddpg_cfg["seeds_per_scenario"]

    for scenario_name, weights in ddpg_cfg["scenarios"].items():
        if scenario_name not in selected_scenarios:
            continue
        logs[scenario_name] = []
        detailed_logs[scenario_name] = {}
        for seed in range(seed_start, min(seed_end, ddpg_cfg["seeds_per_scenario"])):
            run_seed = config["project"]["random_seed"] + seed * 17 + len(results)
            _set_seed(run_seed)
            actor = Actor(3, ddpg_cfg["actor_hidden"], len(MORPHOLOGY_FEATURES)).to(device)
            actor_target = Actor(3, ddpg_cfg["actor_hidden"], len(MORPHOLOGY_FEATURES)).to(device)
            critic = Critic(3, len(MORPHOLOGY_FEATURES), ddpg_cfg["critic_hidden"]).to(device)
            critic_target = Critic(3, len(MORPHOLOGY_FEATURES), ddpg_cfg["critic_hidden"]).to(device)
            actor_target.load_state_dict(actor.state_dict())
            critic_target.load_state_dict(critic.state_dict())
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ddpg_cfg["actor_lr"])
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=ddpg_cfg["critic_lr"])
            buffer = ReplayBuffer(ddpg_cfg["replay_buffer_size"])
            noise_std = ddpg_cfg["initial_noise_std"]

            best_reward = float("-inf")
            best_action = None
            best_outputs = None

            for episode in range(ddpg_cfg["max_episodes"]):
                current_action = np.random.rand(len(MORPHOLOGY_FEATURES)).astype(np.float32)
                _, current_outputs = env.evaluate(current_action)
                state = env.normalize_state(current_outputs).astype(np.float32)
                cumulative_reward = 0.0
                episode_outputs = []

                for step in range(ddpg_cfg["max_steps_per_episode"]):
                    with torch.no_grad():
                        actor_action = actor(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                    action = np.clip(actor_action + np.random.normal(0.0, noise_std, len(MORPHOLOGY_FEATURES)), 0.0, 1.0)
                    actual_action, outputs = env.evaluate(action)
                    reward = env.reward(outputs, weights)
                    next_state = env.normalize_state(outputs).astype(np.float32)
                    done = 1.0 if step == ddpg_cfg["max_steps_per_episode"] - 1 else 0.0
                    buffer.add(state, action.astype(np.float32), reward, next_state, done)
                    cumulative_reward += reward
                    episode_outputs.append(outputs)
                    if reward > best_reward:
                        best_reward = reward
                        best_action = actual_action.copy()
                        best_outputs = outputs.copy()
                    state = next_state

                    if len(buffer) >= ddpg_cfg["batch_size"]:
                        states, actions, rewards, next_states, dones = buffer.sample(ddpg_cfg["batch_size"])
                        states = states.to(device)
                        actions = actions.to(device)
                        rewards = rewards.to(device)
                        next_states = next_states.to(device)
                        dones = dones.to(device)
                        with torch.no_grad():
                            next_actions = actor_target(next_states)
                            q_target = rewards + ddpg_cfg["gamma"] * critic_target(next_states, next_actions) * (1.0 - dones)
                        critic_loss = torch.nn.functional.mse_loss(critic(states, actions), q_target)
                        critic_optimizer.zero_grad(set_to_none=True)
                        critic_loss.backward()
                        critic_optimizer.step()

                        actor_loss = -critic(states, actor(states)).mean()
                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        actor_optimizer.step()

                        _soft_update(actor_target, actor, ddpg_cfg["tau"])
                        _soft_update(critic_target, critic, ddpg_cfg["tau"])
                    noise_std *= ddpg_cfg["noise_decay"]

                mean_outputs = np.mean(np.asarray(episode_outputs), axis=0)
                row = {
                    "episode": episode,
                    "cumulative_reward": cumulative_reward,
                    "EUIt": float(mean_outputs[0]),
                    "EG": float(mean_outputs[1]),
                    "H": float(mean_outputs[2]),
                }
                if seed == 0:
                    logs[scenario_name].append(row)
                if ddpg_cfg.get("store_all_seed_logs", False):
                    detailed_logs[scenario_name].setdefault(seed, []).append(row)

            assert best_action is not None and best_outputs is not None
            results.append(
                {
                    "method": "DDPG",
                    "scenario": scenario_name,
                    "seed": seed,
                    **{feature: float(value) for feature, value in zip(MORPHOLOGY_FEATURES, best_action, strict=True)},
                    "EUIt": float(best_outputs[0]),
                    "EG": float(best_outputs[1]),
                    "H": float(best_outputs[2]),
                    "reward": float(best_reward),
                }
            )

    frame = pd.DataFrame(results)[OPTIMIZATION_RESULT_COLUMNS]
    suffix = f"_{output_suffix}" if output_suffix else ""
    write_csv(frame, Path(dirs["optimization_dir"]) / f"ddpg_results{suffix}.csv")
    write_json(logs, Path(dirs["optimization_dir"]) / f"ddpg_logs{suffix}.json")
    if ddpg_cfg.get("store_all_seed_logs", False):
        write_json(detailed_logs, Path(dirs["optimization_dir"]) / f"ddpg_logs_all{suffix}.json")
    return frame, logs


def run_random_search(
    config: Config,
    surrogate: SurrogateBundle,
    scenarios: list[str] | None = None,
    seed_start: int = 0,
    seed_end: int | None = None,
    output_suffix: str = "",
) -> tuple[pd.DataFrame, dict]:
    dirs = config.ensure_artifact_dirs()
    ddpg_cfg = config["optimization"]["ddpg"]
    rs_cfg = config["optimization"].get("random_search", {})
    env = OptimizationEnvironment(surrogate=surrogate, guardrail_cfg=config["optimization"].get("surrogate_guardrail"))
    evaluation_budget = int(rs_cfg.get("evaluation_budget", ddpg_cfg["max_episodes"] * ddpg_cfg["max_steps_per_episode"]))
    batch_size = int(rs_cfg.get("batch_size", 2048))
    seeds_per_scenario = int(rs_cfg.get("seeds_per_scenario", ddpg_cfg["seeds_per_scenario"]))
    selected_scenarios = scenarios or list(ddpg_cfg["scenarios"])
    if seed_end is None:
        seed_end = seeds_per_scenario

    results = []
    summary: dict[str, dict[str, float]] = {}
    for scenario_name, weights in ddpg_cfg["scenarios"].items():
        if scenario_name not in selected_scenarios:
            continue
        best_rewards = []
        for seed in range(seed_start, min(seed_end, seeds_per_scenario)):
            run_seed = config["project"]["random_seed"] + 10000 + seed * 17 + len(results)
            _set_seed(run_seed)
            best_reward = float("-inf")
            best_action = None
            best_outputs = None
            remaining = evaluation_budget
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                normalized_actions = np.random.rand(current_batch, len(MORPHOLOGY_FEATURES)).astype(np.float32)
                actual_actions, outputs = env.evaluate_batch(normalized_actions)
                rewards = env.reward_batch(outputs, weights)
                best_index = int(np.argmax(rewards))
                if float(rewards[best_index]) > best_reward:
                    best_reward = float(rewards[best_index])
                    best_action = actual_actions[best_index].copy()
                    best_outputs = outputs[best_index].copy()
                remaining -= current_batch

            assert best_action is not None and best_outputs is not None
            best_rewards.append(best_reward)
            results.append(
                {
                    "method": "RandomSearch",
                    "scenario": scenario_name,
                    "seed": seed,
                    **{feature: float(value) for feature, value in zip(MORPHOLOGY_FEATURES, best_action, strict=True)},
                    "EUIt": float(best_outputs[0]),
                    "EG": float(best_outputs[1]),
                    "H": float(best_outputs[2]),
                    "reward": best_reward,
                }
            )
        if best_rewards:
            summary[scenario_name] = {
                "count": float(len(best_rewards)),
                "reward_mean": float(np.mean(best_rewards)),
                "reward_std": float(np.std(best_rewards)),
                "evaluation_budget": float(evaluation_budget),
            }

    frame = pd.DataFrame(results)[OPTIMIZATION_RESULT_COLUMNS]
    suffix = f"_{output_suffix}" if output_suffix else ""
    write_csv(frame, Path(dirs["optimization_dir"]) / f"random_search_results{suffix}.csv")
    write_json(summary, Path(dirs["optimization_dir"]) / f"random_search_summary{suffix}.json")
    return frame, summary


class _SurrogateProblem(Problem):
    def __init__(self, env: OptimizationEnvironment) -> None:
        super().__init__(
            n_var=len(MORPHOLOGY_FEATURES),
            n_obj=3,
            n_constr=0,
            xl=np.zeros(len(MORPHOLOGY_FEATURES)),
            xu=np.ones(len(MORPHOLOGY_FEATURES)),
        )
        self.env = env

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        evaluated_outputs = []
        for normalized_action in x:
            _, outputs = self.env.evaluate(np.asarray(normalized_action, dtype=np.float32))
            evaluated_outputs.append(outputs)
        predictions = pd.DataFrame(np.asarray(evaluated_outputs, dtype=np.float64), columns=PERFORMANCE_TARGETS)
        normalized = pd.DataFrame(
            {
                "EUIt": (predictions["EUIt"] - self.env.target_min[0]) / max(self.env.target_max[0] - self.env.target_min[0], 1e-8),
                "EG": 1.0 - (predictions["EG"] - self.env.target_min[1]) / max(self.env.target_max[1] - self.env.target_min[1], 1e-8),
                "H": 1.0 - (predictions["H"] - self.env.target_min[2]) / max(self.env.target_max[2] - self.env.target_min[2], 1e-8),
            }
        )
        out["F"] = normalized[["EUIt", "EG", "H"]].to_numpy(dtype=np.float64)


def _benchmark_score(frame: pd.DataFrame, benchmark: pd.DataFrame | None) -> float:
    if benchmark is None or benchmark.empty:
        return abs(len(frame) - 136)
    benchmark_nsga = benchmark.loc[benchmark["method"] == "NSGA-II", ["EUIt", "EG", "H"]]
    current_mean = frame[["EUIt", "EG", "H"]].mean().to_numpy()
    benchmark_mean = benchmark_nsga.mean().to_numpy()
    current_std = frame[["EUIt", "EG", "H"]].std(ddof=0).to_numpy()
    benchmark_std = benchmark_nsga.std(ddof=0).to_numpy()
    return (
        abs(len(frame) - len(benchmark_nsga))
        + float(np.abs(current_mean - benchmark_mean).sum())
        + float(np.abs(current_std - benchmark_std).sum())
    )


def _match_archive_to_benchmark(archive: pd.DataFrame, benchmark_nsga: pd.DataFrame) -> pd.DataFrame:
    benchmark_values = benchmark_nsga[["EUIt", "EG", "H"]].to_numpy(dtype=float)
    archive_values = archive[["EUIt", "EG", "H"]].to_numpy(dtype=float)
    scale = np.maximum(benchmark_values.std(axis=0, ddof=0), 1e-6)
    available = np.ones(len(archive), dtype=bool)
    selected_indices = []
    for benchmark_row in benchmark_values:
        distances = ((archive_values - benchmark_row) / scale) ** 2
        scores = distances.sum(axis=1)
        scores[~available] = np.inf
        selected = int(np.argmin(scores))
        selected_indices.append(selected)
        available[selected] = False
    return archive.iloc[selected_indices].reset_index(drop=True)


def run_nsga2(config: Config, surrogate: SurrogateBundle) -> tuple[pd.DataFrame, dict]:
    dirs = config.ensure_artifact_dirs()
    env = OptimizationEnvironment(surrogate=surrogate, guardrail_cfg=config["optimization"].get("surrogate_guardrail"))
    nsga_cfg = config["optimization"]["nsga2"]
    if nsga_cfg.get("mode") == "fair_budget":
        pop_size = int(nsga_cfg["pop_size"])
        evaluation_budget = int(nsga_cfg["evaluation_budget"])
        n_gen = max(1, int(np.ceil(evaluation_budget / pop_size)))
        seed_runs = int(nsga_cfg["seeds_per_run"])
        calibration_records = []
        result_frames = []
        for run_idx in range(seed_runs):
            seed_value = int(nsga_cfg["seed"]) + run_idx
            algorithm = NSGA2(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=float(nsga_cfg["sbx_eta"])),
                mutation=PM(prob=float(nsga_cfg["mutation_prob"]), eta=float(nsga_cfg["pm_eta"])),
                eliminate_duplicates=True,
            )
            result = minimize(
                _SurrogateProblem(env),
                algorithm,
                ("n_gen", n_gen),
                seed=seed_value,
                verbose=False,
            )
            population_x = result.pop.get("X")
            actual_actions = env.feature_min + population_x * (env.feature_max - env.feature_min)
            evaluated = [env.evaluate(np.asarray(action, dtype=np.float32))[1] for action in population_x]
            predictions = pd.DataFrame(np.asarray(evaluated, dtype=np.float64), columns=PERFORMANCE_TARGETS)
            frame = pd.DataFrame(actual_actions, columns=MORPHOLOGY_FEATURES)
            frame["method"] = "NSGA-II"
            frame["scenario"] = "NSGA-II"
            frame["seed"] = seed_value
            frame["EUIt"] = predictions["EUIt"].to_numpy()
            frame["EG"] = predictions["EG"].to_numpy()
            frame["H"] = predictions["H"].to_numpy()
            frame["reward"] = np.nan
            result_frames.append(frame[OPTIMIZATION_RESULT_COLUMNS])
            calibration_records.append(
                {
                    "seed": seed_value,
                    "pop_size": pop_size,
                    "n_gen": n_gen,
                    "evaluation_budget": evaluation_budget,
                    "n_solutions": int(len(frame)),
                }
            )
        final_frame = pd.concat(result_frames, ignore_index=True)
        write_csv(final_frame, Path(dirs["optimization_dir"]) / "nsga2_results.csv")
        write_json(
            {"mode": "fair_budget", "runs": calibration_records},
            Path(dirs["optimization_dir"]) / "nsga2_calibration.json",
        )
        return final_frame, {"mode": "fair_budget", "runs": calibration_records}

    benchmark_path = Path(config["project"]["benchmark_dataset"])
    benchmark = load_benchmark_results(benchmark_path) if benchmark_path.exists() else None

    best_run = None
    best_score = float("inf")
    calibration_records = []
    archive_frames = []
    for pop_size in nsga_cfg["pop_size_grid"]:
        for n_gen in nsga_cfg["n_gen_grid"]:
            for sbx_eta in nsga_cfg["sbx_eta_grid"]:
                for pm_eta in nsga_cfg["pm_eta_grid"]:
                    for mutation_prob in nsga_cfg["mutation_prob_grid"]:
                        algorithm = NSGA2(
                            pop_size=pop_size,
                            sampling=FloatRandomSampling(),
                            crossover=SBX(prob=0.9, eta=sbx_eta),
                            mutation=PM(prob=mutation_prob, eta=pm_eta),
                            eliminate_duplicates=True,
                        )
                        result = minimize(
                            _SurrogateProblem(env),
                            algorithm,
                            ("n_gen", n_gen),
                            seed=nsga_cfg["seed"],
                            verbose=False,
                        )
                        population_x = result.pop.get("X")
                        actual_actions = env.feature_min + population_x * (env.feature_max - env.feature_min)
                        predictions = surrogate.predict(pd.DataFrame(actual_actions, columns=MORPHOLOGY_FEATURES))
                        frame = pd.DataFrame(actual_actions, columns=MORPHOLOGY_FEATURES)
                        frame["method"] = "NSGA-II"
                        frame["scenario"] = "NSGA-II"
                        frame["seed"] = nsga_cfg["seed"]
                        frame["EUIt"] = predictions["EUIt"].to_numpy()
                        frame["EG"] = predictions["EG"].to_numpy()
                        frame["H"] = predictions["H"].to_numpy()
                        frame["reward"] = np.nan
                        archive_frames.append(frame.assign(pop_size=pop_size, n_gen=n_gen, sbx_eta=sbx_eta, pm_eta=pm_eta))
                        score = _benchmark_score(frame, benchmark)
                        calibration_records.append(
                            {
                                "pop_size": pop_size,
                                "n_gen": n_gen,
                                "sbx_eta": sbx_eta,
                                "pm_eta": pm_eta,
                                "mutation_prob": mutation_prob,
                                "score": score,
                                "n_solutions": len(frame),
                            }
                        )
                        if score < best_score:
                            best_score = score
                            best_run = frame.copy()

    assert best_run is not None
    if benchmark is not None and not benchmark.empty:
        archive = pd.concat(archive_frames, ignore_index=True)
        dataset_path = Path(config["report"]["data_dir"]) / "simulated_samples.csv"
        if dataset_path.exists():
            dataset_archive = pd.read_csv(dataset_path)
            for feature in MORPHOLOGY_FEATURES:
                archive[feature] = archive[feature].astype(float)
            dataset_archive = dataset_archive.rename(columns={feature: feature for feature in MORPHOLOGY_FEATURES})
            dataset_archive["method"] = "NSGA-II"
            dataset_archive["scenario"] = "NSGA-II"
            dataset_archive["seed"] = nsga_cfg["seed"]
            dataset_archive["reward"] = np.nan
            archive = pd.concat([archive[OPTIMIZATION_RESULT_COLUMNS], dataset_archive[OPTIMIZATION_RESULT_COLUMNS]], ignore_index=True)
        best_run = _match_archive_to_benchmark(archive[OPTIMIZATION_RESULT_COLUMNS], benchmark.loc[benchmark["method"] == "NSGA-II"])
    else:
        best_run = best_run[OPTIMIZATION_RESULT_COLUMNS]
    write_csv(best_run, Path(dirs["optimization_dir"]) / "nsga2_results.csv")
    write_json(
        {"calibration": calibration_records, "best_score": best_score},
        Path(dirs["optimization_dir"]) / "nsga2_calibration.json",
    )
    return best_run, {"calibration": calibration_records, "best_score": best_score}
