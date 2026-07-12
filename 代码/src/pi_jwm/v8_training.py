"""Training helpers for PI-JWM v8 experiments."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pi_jwm.v6_data import inverse_normalize, inverse_transform_link_rate
from pi_jwm.v6_dual_graph import V6DualGraphBatch
from pi_jwm.v6_metrics import active_rate_metrics, activity_metrics, regression_metrics
from pi_jwm.v8_full_world_model import V8FullWorldModelConfig, V8FullWorldModelRollout


def build_v8_model_from_arrays(
    arrays: dict[str, np.ndarray],
    hidden_dim: int = 64,
    graph_mode: str = "dual",
    fusion_mode: str = "gated",
    fusion_num_heads: int = 4,
    active_rate_auxiliary: bool = False,
    active_rate_head_mode: str = "mlp",
    num_rate_experts: int = 4,
    rate_output_mode: str = "direct",
    history_encoder: str = "mean",
    latent_transition_mode: str = "message_passing",
    activity_memory_dim: int = 0,
    activity_memory_routing: str = "none",
    adaptive_edge_context: str = "none",
    adaptive_edge_topk: int = 8,
    return_message_diagnostics: bool = False,
) -> V8FullWorldModelRollout:
    config = V8FullWorldModelConfig(
        node_dim=int(arrays["x_node"].shape[-1]),
        physical_edge_dim=8,
        info_edge_dim=int(arrays["x_link"].shape[-1]),
        action_dim=int(arrays["edge_a_hist"].shape[-1]),
        task_dim=int(arrays["x_task"].shape[-1]),
        hidden_dim=hidden_dim,
        horizon=int(arrays["edge_a_future"].shape[1]),
        graph_mode=graph_mode,
        fusion_mode=fusion_mode,
        fusion_num_heads=fusion_num_heads,
        active_rate_auxiliary=active_rate_auxiliary,
        active_rate_head_mode=active_rate_head_mode,
        num_rate_experts=num_rate_experts,
        rate_output_mode=rate_output_mode,
        history_encoder=history_encoder,
        latent_transition_mode=latent_transition_mode,
        activity_memory_dim=activity_memory_dim,
        activity_memory_routing=activity_memory_routing,
        adaptive_edge_context=adaptive_edge_context,
        adaptive_edge_topk=adaptive_edge_topk,
        edge_src_idx=torch.as_tensor(arrays["edge_src_idx"], dtype=torch.long),
        edge_dst_idx=torch.as_tensor(arrays["edge_dst_idx"], dtype=torch.long),
        return_message_diagnostics=return_message_diagnostics,
    )
    return V8FullWorldModelRollout(config)


def train_v8_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rate_loss_mode: str = "weighted_all",
    inactive_rate_weight: float = 0.05,
    active_rate_auxiliary_weight: float = 0.0,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
    node_loss_weight: float = 0.5,
    activity_loss_weight: float = 1.0,
    rate_loss_weight: float = 0.3,
    task_loss_weight: float = 0.8,
    activity_loss_mode: str = "bce",
    activity_pos_weight: float = 80.0,
    activity_focal_gamma: float = 2.0,
    inactive_loss_sample_ratio: float = 1.0,
    false_positive_penalty_weight: float = 0.0,
    dynamic_hard_negative_weight: float = 0.0,
    dynamic_hard_negative_ratio: float = 0.1,
    hurdle_train_gate_mode: str = "predicted",
    hurdle_train_gate_power: float = 1.0,
    positive_rate_specialist_weight: float = 0.0,
    positive_rate_target_mode: str = "raw",
    positive_rate_loss_mode: str = "mse",
    positive_rate_tweedie_power: float = 1.5,
    positive_rate_raw_stats: tuple[np.ndarray, np.ndarray] | None = None,
    high_rate_weight: float = 1.0,
    high_rate_threshold: float = 0.0,
    active_rate_reweight_mode: str = "none",
    active_rate_lds_config: dict[str, object] | None = None,
    active_rate_bmc_noise_sigma: float = 1.0,
    active_rate_bmc_minimum_count: int = 3,
    active_mass_loss_weight: float = 0.0,
    active_mass_target_mode: str = "normalized",
    active_mass_raw_stats: tuple[np.ndarray, np.ndarray] | None = None,
    candidate_loss_mask: torch.Tensor | np.ndarray | None = None,
    candidate_rate_loss_mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    model.train()
    rows = []
    for batch, target in loader:
        batch = move_v8_batch_to_device(batch, device)
        target = move_v8_target_to_device(target, device)
        outputs = model(batch)
        loss, parts = compute_v8_loss(
            outputs,
            target,
            rate_loss_mode=rate_loss_mode,
            inactive_rate_weight=inactive_rate_weight,
            active_rate_auxiliary_weight=active_rate_auxiliary_weight,
            rate_output_mode=rate_output_mode,
            inactive_rate_value=inactive_rate_value,
            node_loss_weight=node_loss_weight,
            activity_loss_weight=activity_loss_weight,
            rate_loss_weight=rate_loss_weight,
            task_loss_weight=task_loss_weight,
            activity_loss_mode=activity_loss_mode,
            activity_pos_weight=activity_pos_weight,
            activity_focal_gamma=activity_focal_gamma,
            inactive_loss_sample_ratio=inactive_loss_sample_ratio,
            false_positive_penalty_weight=false_positive_penalty_weight,
            dynamic_hard_negative_weight=dynamic_hard_negative_weight,
            dynamic_hard_negative_ratio=dynamic_hard_negative_ratio,
            hurdle_train_gate_mode=hurdle_train_gate_mode,
            hurdle_train_gate_power=hurdle_train_gate_power,
            positive_rate_specialist_weight=positive_rate_specialist_weight,
            positive_rate_target_mode=positive_rate_target_mode,
            positive_rate_loss_mode=positive_rate_loss_mode,
            positive_rate_tweedie_power=positive_rate_tweedie_power,
            positive_rate_raw_stats=positive_rate_raw_stats,
            high_rate_weight=high_rate_weight,
            high_rate_threshold=high_rate_threshold,
            active_rate_reweight_mode=active_rate_reweight_mode,
            active_rate_lds_config=active_rate_lds_config,
            active_rate_bmc_noise_sigma=active_rate_bmc_noise_sigma,
            active_rate_bmc_minimum_count=active_rate_bmc_minimum_count,
            active_mass_loss_weight=active_mass_loss_weight,
            active_mass_target_mode=active_mass_target_mode,
            active_mass_raw_stats=active_mass_raw_stats,
            candidate_loss_mask=candidate_loss_mask,
            candidate_rate_loss_mask=candidate_rate_loss_mask,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        rows.append(parts)
    return mean_metric_rows(rows)


def evaluate_v8_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: dict,
    activity_threshold: float | None = None,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
    candidate_eval_mask: np.ndarray | torch.Tensor | None = None,
    hurdle_gate_temperature: float = 1.0,
    hurdle_gate_power: float = 1.0,
) -> dict[str, dict[str, float]]:
    predictions = collect_v8_predictions(
        model,
        loader,
        device,
        stats,
        rate_output_mode=rate_output_mode,
        inactive_rate_value=inactive_rate_value,
        hurdle_gate_temperature=hurdle_gate_temperature,
        hurdle_gate_power=hurdle_gate_power,
    )
    if candidate_eval_mask is not None:
        predictions = apply_candidate_eval_mask(predictions, candidate_eval_mask)
    threshold = (
        choose_activity_threshold(predictions["link_activity_prob"], predictions["link_activity_true"])
        if activity_threshold is None
        else activity_threshold
    )
    metrics = {
        "node": regression_metrics(predictions["node_pred"], predictions["node_true"]),
        "task": regression_metrics(predictions["task_pred"], predictions["task_true"]),
        "link_rate": regression_metrics(predictions["link_rate_pred"], predictions["link_rate_true"]),
        "active_rate": active_rate_metrics(
            predictions["link_rate_pred"],
            predictions["link_rate_true"],
            predictions["link_activity_true"],
        ),
        "activity": activity_metrics(
            predictions["link_activity_prob"],
            predictions["link_activity_true"],
            threshold=threshold,
        ),
    }
    if "link_active_rate_aux_pred" in predictions:
        metrics["active_rate_auxiliary"] = active_rate_metrics(
            predictions["link_active_rate_aux_pred"],
            predictions["link_rate_true"],
            predictions["link_activity_true"],
        )
    if "link_positive_rate_pred" in predictions:
        metrics["positive_rate_active"] = active_rate_metrics(
            predictions["link_positive_rate_pred"],
            predictions["link_rate_true"],
            predictions["link_activity_true"],
        )
        metrics["hurdle_gate"] = {
            "temperature": float(hurdle_gate_temperature),
            "power": float(hurdle_gate_power),
        }
    if "link_active_mass_rate_pred" in predictions:
        metrics["active_mass"] = active_rate_metrics(
            predictions["link_active_mass_rate_pred"],
            predictions["link_rate_true"],
            predictions["link_activity_true"],
        )
    return metrics


def collect_v8_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: dict,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
    hurdle_gate_temperature: float = 1.0,
    hurdle_gate_power: float = 1.0,
) -> dict[str, np.ndarray]:
    if hurdle_gate_temperature <= 0.0:
        raise ValueError("hurdle_gate_temperature must be positive")
    if hurdle_gate_power < 0.0:
        raise ValueError("hurdle_gate_power must be non-negative")
    model.eval()
    rows = {
        "node_pred": [],
        "node_true": [],
        "link_activity_prob": [],
        "link_activity_true": [],
        "link_rate_pred": [],
        "link_positive_rate_pred": [],
        "link_active_rate_aux_pred": [],
        "link_active_mass_rate_pred": [],
        "link_rate_true": [],
        "task_pred": [],
        "task_true": [],
    }
    with torch.no_grad():
        for batch, target in loader:
            batch_device = move_v8_batch_to_device(batch, device)
            outputs = model(batch_device)
            target_device = move_v8_target_to_device(target, device)
            selected_link_rate = select_v8_link_rate_output(
                outputs,
                target=target_device,
                rate_output_mode=rate_output_mode,
                inactive_rate_value=inactive_rate_value,
            )
            rows["node_pred"].append(inverse_normalize(outputs["node"].cpu().numpy(), stats["y_node"]))
            rows["node_true"].append(inverse_normalize(target["node"].numpy(), stats["y_node"]))
            rows["link_activity_prob"].append(torch.sigmoid(outputs["link_activity_logit"]).cpu().numpy())
            rows["link_activity_true"].append(target["link_activity"].numpy())
            rows["link_rate_pred"].append(
                denormalize_v8_link_rate_prediction(
                    outputs["link_rate"].cpu().numpy(),
                    stats,
                    baseline=(
                        batch.link_rate_baseline.cpu().numpy()
                        if batch.link_rate_baseline is not None
                        else None
                    ),
                )
            )
            if rate_output_mode != "main":
                rows["link_rate_pred"][-1] = denormalize_v8_link_rate_prediction(
                    selected_link_rate.cpu().numpy(),
                    stats,
                    baseline=(
                        batch.link_rate_baseline.cpu().numpy()
                        if batch.link_rate_baseline is not None
                        else None
                    ),
                )
            if "link_active_rate_aux" in outputs:
                rows["link_active_rate_aux_pred"].append(
                    denormalize_v8_link_rate_prediction(
                        outputs["link_active_rate_aux"].cpu().numpy(),
                        stats,
                        baseline=(
                            batch.link_rate_baseline.cpu().numpy()
                            if batch.link_rate_baseline is not None
                            else None
                        ),
                    )
                )
            if "link_positive_rate" in outputs:
                positive_rate_norm = outputs["link_positive_rate"]
                if hurdle_gate_temperature != 1.0 or hurdle_gate_power != 1.0:
                    gate = calibrate_hurdle_gate_tensor(
                        torch.sigmoid(outputs["link_activity_logit"]),
                        temperature=hurdle_gate_temperature,
                        power=hurdle_gate_power,
                    )
                    rows["link_rate_pred"][-1] = denormalize_v8_link_rate_prediction(
                        (gate * positive_rate_norm).cpu().numpy(),
                        stats,
                        baseline=(
                            batch.link_rate_baseline.cpu().numpy()
                            if batch.link_rate_baseline is not None
                            else None
                        ),
                    )
                rows["link_positive_rate_pred"].append(
                    denormalize_v8_link_rate_prediction(
                        positive_rate_norm.cpu().numpy(),
                        stats,
                        baseline=(
                            batch.link_rate_baseline.cpu().numpy()
                            if batch.link_rate_baseline is not None
                            else None
                        ),
                    )
                )
            if "link_active_mass_rate" in outputs:
                rows["link_active_mass_rate_pred"].append(
                    denormalize_v8_link_rate_prediction(
                        outputs["link_active_mass_rate"].cpu().numpy(),
                        stats,
                        baseline=(
                            batch.link_rate_baseline.cpu().numpy()
                            if batch.link_rate_baseline is not None
                            else None
                        ),
                    )
                )
            rows["link_rate_true"].append(
                denormalize_v8_link_rate_prediction(
                    target["link_rate"].numpy(),
                    stats,
                    baseline=(
                        batch.link_rate_baseline.cpu().numpy()
                        if batch.link_rate_baseline is not None
                        else None
                    ),
                    clip_prediction=False,
                )
            )
            rows["task_pred"].append(inverse_normalize(outputs["task"].cpu().numpy(), stats["y_task"]))
            rows["task_true"].append(inverse_normalize(target["task"].numpy(), stats["y_task"]))
    return {name: np.concatenate(values, axis=0) for name, values in rows.items() if values}


def calibrate_hurdle_gate_tensor(prob: torch.Tensor, temperature: float = 1.0, power: float = 1.0) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("hurdle_gate_temperature must be positive")
    if power < 0.0:
        raise ValueError("hurdle_gate_power must be non-negative")
    clipped = prob.clamp(1e-6, 1.0 - 1e-6)
    if power == 0.0:
        return torch.ones_like(clipped)
    logits = torch.logit(clipped)
    calibrated = torch.sigmoid(logits / float(temperature))
    return calibrated.pow(float(power)).clamp(0.0, 1.0)


def move_v8_batch_to_device(batch: V6DualGraphBatch, device: torch.device) -> V6DualGraphBatch:
    return V6DualGraphBatch(
        node_history=batch.node_history.to(device),
        physical_edge_history=batch.physical_edge_history.to(device),
        info_edge_history=batch.info_edge_history.to(device),
        action_history=batch.action_history.to(device),
        future_actions=batch.future_actions.to(device),
        task_history=batch.task_history.to(device),
        link_rate_baseline=(
            batch.link_rate_baseline.to(device) if batch.link_rate_baseline is not None else None
        ),
    )


def move_v8_target_to_device(target: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in target.items()}


def compute_v8_loss(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    rate_loss_mode: str = "weighted_all",
    inactive_rate_weight: float = 0.05,
    active_rate_auxiliary_weight: float = 0.0,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
    node_loss_weight: float = 0.5,
    activity_loss_weight: float = 1.0,
    rate_loss_weight: float = 0.3,
    task_loss_weight: float = 0.8,
    activity_loss_mode: str = "bce",
    activity_pos_weight: float = 80.0,
    activity_focal_gamma: float = 2.0,
    inactive_loss_sample_ratio: float = 1.0,
    false_positive_penalty_weight: float = 0.0,
    dynamic_hard_negative_weight: float = 0.0,
    dynamic_hard_negative_ratio: float = 0.1,
    hurdle_train_gate_mode: str = "predicted",
    hurdle_train_gate_power: float = 1.0,
    positive_rate_specialist_weight: float = 0.0,
    positive_rate_target_mode: str = "raw",
    positive_rate_loss_mode: str = "mse",
    positive_rate_tweedie_power: float = 1.5,
    positive_rate_raw_stats: tuple[np.ndarray, np.ndarray] | None = None,
    high_rate_weight: float = 1.0,
    high_rate_threshold: float = 0.0,
    active_rate_reweight_mode: str = "none",
    active_rate_lds_config: dict[str, object] | None = None,
    active_rate_bmc_noise_sigma: float = 1.0,
    active_rate_bmc_minimum_count: int = 3,
    active_mass_loss_weight: float = 0.0,
    active_mass_target_mode: str = "normalized",
    active_mass_raw_stats: tuple[np.ndarray, np.ndarray] | None = None,
    loss_sampling_seed: int | None = None,
    candidate_loss_mask: torch.Tensor | np.ndarray | None = None,
    candidate_rate_loss_mask: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if active_rate_auxiliary_weight < 0.0:
        raise ValueError("active_rate_auxiliary_weight must be non-negative")
    for name, value in {
        "node_loss_weight": node_loss_weight,
        "activity_loss_weight": activity_loss_weight,
        "rate_loss_weight": rate_loss_weight,
        "task_loss_weight": task_loss_weight,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if activity_pos_weight <= 0.0:
        raise ValueError("activity_pos_weight must be positive")
    if activity_focal_gamma < 0.0:
        raise ValueError("activity_focal_gamma must be non-negative")
    if inactive_loss_sample_ratio <= 0.0 or inactive_loss_sample_ratio > 1.0:
        raise ValueError("inactive_loss_sample_ratio must be in (0, 1]")
    if false_positive_penalty_weight < 0.0:
        raise ValueError("false_positive_penalty_weight must be non-negative")
    if dynamic_hard_negative_weight < 0.0:
        raise ValueError("dynamic_hard_negative_weight must be non-negative")
    if dynamic_hard_negative_ratio <= 0.0 or dynamic_hard_negative_ratio > 1.0:
        raise ValueError("dynamic_hard_negative_ratio must be in (0, 1]")
    if hurdle_train_gate_mode not in {"none", "predicted", "detach", "teacher_forcing"}:
        raise ValueError("hurdle_train_gate_mode must be one of: none, predicted, detach, teacher_forcing")
    if hurdle_train_gate_power < 0.0:
        raise ValueError("hurdle_train_gate_power must be non-negative")
    if positive_rate_specialist_weight < 0.0:
        raise ValueError("positive_rate_specialist_weight must be non-negative")
    if positive_rate_target_mode not in {"raw", "log1p", "normalized", "log1p_normalized"}:
        raise ValueError("positive_rate_target_mode must be one of: raw, log1p, normalized, log1p_normalized")
    if positive_rate_loss_mode not in {"mse", "huber", "tweedie"}:
        raise ValueError("positive_rate_loss_mode must be one of: mse, huber, tweedie")
    if positive_rate_tweedie_power <= 1.0 or positive_rate_tweedie_power >= 2.0:
        raise ValueError("positive_rate_tweedie_power must be in (1, 2)")
    if high_rate_weight < 1.0:
        raise ValueError("high_rate_weight must be at least 1.0")
    if high_rate_threshold < 0.0:
        raise ValueError("high_rate_threshold must be non-negative")
    if active_rate_reweight_mode not in {"none", "lds", "bmc"}:
        raise ValueError("active_rate_reweight_mode must be one of: none, lds, bmc")
    if active_rate_reweight_mode == "lds" and active_rate_lds_config is None:
        raise ValueError("active_rate_reweight_mode='lds' requires active_rate_lds_config")
    if active_rate_bmc_noise_sigma <= 0.0:
        raise ValueError("active_rate_bmc_noise_sigma must be positive")
    if active_rate_bmc_minimum_count < 2:
        raise ValueError("active_rate_bmc_minimum_count must be at least 2")
    if active_mass_loss_weight < 0.0:
        raise ValueError("active_mass_loss_weight must be non-negative")
    if active_mass_target_mode not in {"normalized", "raw"}:
        raise ValueError("active_mass_target_mode must be one of: normalized, raw")
    candidate_mask = expand_candidate_loss_mask(candidate_loss_mask, target["link_activity"])
    candidate_rate_mask = expand_candidate_loss_mask(candidate_rate_loss_mask, target["link_activity"])
    if candidate_rate_mask is None:
        candidate_rate_mask = candidate_mask
    mse = nn.MSELoss()
    node_loss = mse(outputs["node"], target["node"])
    activity_loss = compute_activity_loss(
        outputs["link_activity_logit"],
        target["link_activity"],
        mode=activity_loss_mode,
        pos_weight=activity_pos_weight,
        focal_gamma=activity_focal_gamma,
        loss_mask=candidate_mask,
    )
    selected_rate = select_v8_link_rate_output(
        outputs,
        target=target,
        rate_output_mode=rate_output_mode,
        inactive_rate_value=inactive_rate_value,
    )
    if "link_positive_rate" in outputs:
        selected_rate = select_hurdle_training_rate_output(
            outputs,
            target,
            selected_rate,
            hurdle_train_gate_mode,
            hurdle_train_gate_power,
        )
    rate_error = (selected_rate - target["link_rate"]) ** 2
    active_mask = target["link_activity"] > 0.5
    if candidate_rate_mask is not None:
        active_mask = active_mask & candidate_rate_mask
    inactive_all_mask = target["link_activity"] <= 0.5
    if candidate_rate_mask is not None:
        inactive_all_mask = inactive_all_mask & candidate_rate_mask
    inactive_mask = sample_inactive_loss_mask(inactive_all_mask, inactive_loss_sample_ratio, seed=loss_sampling_seed)
    active_rate_lds_mean_weight = 0.0
    active_rate_bmc_count = 0
    if active_rate_reweight_mode == "lds":
        if "link_rate_raw" not in target:
            raise ValueError("active_rate_reweight_mode='lds' requires target link_rate_raw")
        active_rate_loss, active_rate_lds_mean_weight = compute_lds_weighted_active_loss(
            rate_error,
            target["link_rate_raw"],
            active_mask,
            active_rate_lds_config,
        )
        high_rate_count = 0.0
    elif active_rate_reweight_mode == "bmc":
        active_rate_loss, active_rate_bmc_count = compute_balanced_mse_loss(
            selected_rate[active_mask],
            target["link_rate"][active_mask],
            noise_sigma=active_rate_bmc_noise_sigma,
            minimum_count=active_rate_bmc_minimum_count,
        )
        high_rate_count = 0.0
    else:
        active_rate_loss, high_rate_count = compute_high_rate_weighted_active_loss(
            rate_error,
            target,
            active_mask,
            high_rate_weight=high_rate_weight,
            high_rate_threshold=high_rate_threshold,
        )
    inactive_rate_loss = rate_error[inactive_mask].mean() if inactive_mask.any() else rate_error.new_tensor(0.0)
    activity_prob = torch.sigmoid(outputs["link_activity_logit"])
    false_positive_mask = target["link_activity"] <= 0.5
    if candidate_mask is not None:
        false_positive_mask = false_positive_mask & candidate_mask
    false_positive_penalty = (
        activity_prob[false_positive_mask].pow(2).mean() if false_positive_mask.any() else rate_error.new_tensor(0.0)
    )
    dynamic_hard_negative_activity, dynamic_hard_negative_count = compute_dynamic_hard_negative_activity_loss(
        outputs["link_activity_logit"],
        target["link_activity"],
        ratio=dynamic_hard_negative_ratio,
        loss_mask=candidate_mask,
    )
    if rate_loss_mode == "weighted_all":
        active_weight = 1.0 + 50.0 * target["link_activity"]
        rate_loss = masked_mean(rate_error * active_weight, candidate_rate_mask)
    elif rate_loss_mode == "active_only":
        rate_loss = active_rate_loss
    elif rate_loss_mode == "active_mixed":
        rate_loss = active_rate_loss + inactive_rate_weight * inactive_rate_loss
    else:
        raise ValueError("rate_loss_mode must be one of: weighted_all, active_only, active_mixed")
    task_loss = mse(outputs["task"], target["task"])
    active_rate_auxiliary_loss = rate_error.new_tensor(0.0)
    if active_rate_auxiliary_weight > 0.0 and "link_active_rate_aux" in outputs:
        auxiliary_error = (outputs["link_active_rate_aux"] - target["link_rate"]) ** 2
        if active_mask.any():
            active_rate_auxiliary_loss = auxiliary_error[active_mask].mean()
    positive_rate_specialist_loss = rate_error.new_tensor(0.0)
    if positive_rate_specialist_weight > 0.0:
        if "link_positive_rate" not in outputs:
            raise ValueError("positive_rate_specialist_weight requires link_positive_rate outputs")
        positive_target = target["link_rate_raw"] if positive_rate_loss_mode == "tweedie" and "link_rate_raw" in target else target["link_rate"]
        positive_pred = outputs["link_positive_rate"]
        if positive_rate_loss_mode == "tweedie":
            if positive_rate_raw_stats is None:
                raise ValueError("positive_rate_loss_mode='tweedie' requires positive_rate_raw_stats")
            positive_pred = inverse_normalize_tensor(positive_pred, positive_rate_raw_stats)
        elif positive_rate_target_mode in {"log1p", "log1p_normalized"}:
            positive_target = torch.log1p(torch.clamp(positive_target, min=0.0))
            positive_pred = torch.log1p(torch.clamp(positive_pred, min=0.0))
        positive_mask = active_mask
        if positive_mask.any():
            if positive_rate_loss_mode == "mse":
                positive_rate_specialist_loss = ((positive_pred - positive_target) ** 2)[positive_mask].mean()
            elif positive_rate_loss_mode == "huber":
                positive_rate_specialist_loss = nn.functional.huber_loss(
                    positive_pred[positive_mask],
                    positive_target[positive_mask],
                    reduction="mean",
                )
            else:
                positive_rate_specialist_loss = compute_tweedie_deviance_loss(
                    positive_pred[positive_mask],
                    positive_target[positive_mask],
                    power=positive_rate_tweedie_power,
                )
    active_mass_total_loss = rate_error.new_tensor(0.0)
    if active_mass_loss_weight > 0.0:
        if "link_active_mass_total" not in outputs:
            raise ValueError("active_mass_loss_weight requires link_active_mass_total outputs")
        active_mass_total_loss = compute_active_mass_total_loss(
            outputs["link_active_mass_total"],
            target,
            active_mask,
            target_mode=active_mass_target_mode,
            raw_stats=active_mass_raw_stats,
        )
    node_weighted = node_loss_weight * node_loss
    activity_weighted = (
        activity_loss_weight * (activity_loss + false_positive_penalty_weight * false_positive_penalty)
        + dynamic_hard_negative_weight * dynamic_hard_negative_activity
    )
    rate_weighted = rate_loss_weight * (rate_loss + active_rate_auxiliary_weight * active_rate_auxiliary_loss)
    positive_rate_weighted = positive_rate_specialist_weight * positive_rate_specialist_loss
    active_mass_weighted = active_mass_loss_weight * active_mass_total_loss
    task_weighted = task_loss_weight * task_loss
    total = node_weighted + activity_weighted + rate_weighted + positive_rate_weighted + active_mass_weighted + task_weighted
    return total, {
        "total": float(total.detach().cpu()),
        "node": float(node_loss.detach().cpu()),
        "activity": float(activity_loss.detach().cpu()),
        "rate": float(rate_loss.detach().cpu()),
        "active_rate_auxiliary": float(active_rate_auxiliary_loss.detach().cpu()),
        "active_rate_loss": float(active_rate_loss.detach().cpu()),
        "inactive_rate_loss": float(inactive_rate_loss.detach().cpu()),
        "task": float(task_loss.detach().cpu()),
        "activity_loss_mode": activity_loss_mode,
        "activity_pos_weight": float(activity_pos_weight),
        "false_positive_penalty": float(false_positive_penalty.detach().cpu()),
        "false_positive_penalty_weight": float(false_positive_penalty_weight),
        "dynamic_hard_negative_activity": float(dynamic_hard_negative_activity.detach().cpu()),
        "dynamic_hard_negative_weight": float(dynamic_hard_negative_weight),
        "dynamic_hard_negative_ratio": float(dynamic_hard_negative_ratio),
        "dynamic_hard_negative_count": float(dynamic_hard_negative_count),
        "positive_rate_specialist": float(positive_rate_specialist_loss.detach().cpu()),
        "positive_rate_specialist_weight": float(positive_rate_specialist_weight),
        "positive_rate_target_mode": positive_rate_target_mode,
        "positive_rate_loss_mode": positive_rate_loss_mode,
        "positive_rate_tweedie_power": float(positive_rate_tweedie_power),
        "high_rate_weight": float(high_rate_weight),
        "high_rate_threshold": float(high_rate_threshold),
        "high_rate_count": float(high_rate_count),
        "active_rate_reweight_mode": active_rate_reweight_mode,
        "active_rate_lds_mean_weight": float(active_rate_lds_mean_weight),
        "active_rate_bmc_count": float(active_rate_bmc_count),
        "active_rate_bmc_noise_sigma": float(active_rate_bmc_noise_sigma),
        "active_rate_bmc_minimum_count": float(active_rate_bmc_minimum_count),
        "active_mass_total_loss": float(active_mass_total_loss.detach().cpu()),
        "active_mass_loss_weight": float(active_mass_loss_weight),
        "active_mass_target_mode": active_mass_target_mode,
        "inactive_loss_sample_ratio": float(inactive_loss_sample_ratio),
        "hurdle_train_gate_mode": hurdle_train_gate_mode,
        "hurdle_train_gate_power": float(hurdle_train_gate_power),
        "candidate_loss_edge_count": float(candidate_mask_edge_count(candidate_loss_mask, target["link_activity"])),
        "candidate_rate_loss_edge_count": float(candidate_mask_edge_count(candidate_rate_loss_mask if candidate_rate_loss_mask is not None else candidate_loss_mask, target["link_activity"])),
        "active_loss_count": float(active_mask.sum().detach().cpu()),
        "inactive_loss_count": float(inactive_mask.sum().detach().cpu()),
        "node_weighted": float(node_weighted.detach().cpu()),
        "activity_weighted": float(activity_weighted.detach().cpu()),
        "rate_weighted": float(rate_weighted.detach().cpu()),
        "active_mass_weighted": float(active_mass_weighted.detach().cpu()),
        "task_weighted": float(task_weighted.detach().cpu()),
    }


def compute_high_rate_weighted_active_loss(
    rate_error: torch.Tensor,
    target: dict[str, torch.Tensor],
    active_mask: torch.Tensor,
    high_rate_weight: float = 1.0,
    high_rate_threshold: float = 0.0,
) -> tuple[torch.Tensor, float]:
    if not active_mask.any():
        return rate_error.new_tensor(0.0), 0.0
    active_error = rate_error[active_mask]
    if high_rate_weight == 1.0:
        return active_error.mean(), 0.0
    rate_for_threshold = target.get("link_rate_raw", target["link_rate"])
    high_rate_mask = active_mask & (rate_for_threshold >= high_rate_threshold)
    high_rate_count = float(high_rate_mask.sum().detach().cpu())
    if high_rate_count == 0.0:
        return active_error.mean(), 0.0
    weights = torch.ones_like(rate_error)
    weights = torch.where(high_rate_mask, weights * float(high_rate_weight), weights)
    active_weights = weights[active_mask]
    return (active_error * active_weights).sum() / active_weights.sum().clamp_min(1e-12), high_rate_count


def fit_lds_rate_reweighting(
    active_raw_rates: np.ndarray,
    bin_width: float = 50.0,
    kernel_size: int = 5,
    sigma: float = 2.0,
    weight_min: float = 0.5,
    weight_max: float = 3.0,
    tail_quantile: float = 0.995,
) -> dict[str, object]:
    rates = np.asarray(active_raw_rates, dtype=np.float64).reshape(-1)
    rates = rates[np.isfinite(rates) & (rates > 0.0)]
    if rates.size == 0:
        raise ValueError("active_raw_rates must contain at least one finite positive value")
    if bin_width <= 0.0:
        raise ValueError("bin_width must be positive")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if weight_min <= 0.0 or weight_max < weight_min:
        raise ValueError("weight bounds must satisfy 0 < weight_min <= weight_max")
    if tail_quantile <= 0.0 or tail_quantile > 1.0:
        raise ValueError("tail_quantile must be in (0, 1]")

    tail_cap = max(float(np.quantile(rates, tail_quantile)), float(bin_width))
    bin_count = max(1, int(np.ceil(tail_cap / bin_width)))
    upper_bounds = np.arange(1, bin_count, dtype=np.float64) * float(bin_width)
    bin_indices = np.searchsorted(upper_bounds, rates, side="right")
    empirical_counts = np.bincount(bin_indices, minlength=bin_count).astype(np.float64)

    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel /= kernel.sum()
    padded_counts = np.pad(empirical_counts, (radius, radius), mode="constant")
    effective_density = np.convolve(padded_counts, kernel, mode="valid")
    inverse_density = 1.0 / np.maximum(effective_density, 1e-12)
    sampled_mean = float((inverse_density * empirical_counts).sum() / empirical_counts.sum())
    bin_weights = np.clip(inverse_density / sampled_mean, weight_min, weight_max)

    return {
        "bin_upper_bounds": upper_bounds.astype(float).tolist(),
        "bin_weights": bin_weights.astype(float).tolist(),
        "empirical_counts": empirical_counts.astype(int).tolist(),
        "effective_density": effective_density.astype(float).tolist(),
        "bin_width": float(bin_width),
        "kernel_size": int(kernel_size),
        "sigma": float(sigma),
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
        "tail_quantile": float(tail_quantile),
        "tail_cap": float(tail_cap),
        "active_sample_count": int(rates.size),
    }


def lookup_lds_rate_weights(raw_targets: torch.Tensor, config: dict[str, object]) -> torch.Tensor:
    boundaries = torch.as_tensor(
        config["bin_upper_bounds"],
        dtype=raw_targets.dtype,
        device=raw_targets.device,
    )
    weights = torch.as_tensor(
        config["bin_weights"],
        dtype=raw_targets.dtype,
        device=raw_targets.device,
    )
    if weights.numel() != boundaries.numel() + 1:
        raise ValueError("LDS bin_weights must have exactly one more entry than bin_upper_bounds")
    return weights[torch.bucketize(raw_targets.contiguous(), boundaries)]


def compute_lds_weighted_active_loss(
    rate_error: torch.Tensor,
    raw_target: torch.Tensor,
    active_mask: torch.Tensor,
    config: dict[str, object],
) -> tuple[torch.Tensor, float]:
    if not active_mask.any():
        return rate_error.new_tensor(0.0), 0.0
    active_weights = lookup_lds_rate_weights(raw_target[active_mask], config)
    loss = (rate_error[active_mask] * active_weights).sum() / active_weights.sum().clamp_min(1e-12)
    return loss, float(active_weights.mean().detach().cpu())


def compute_balanced_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    noise_sigma: float = 1.0,
    minimum_count: int = 3,
) -> tuple[torch.Tensor, int]:
    if noise_sigma <= 0.0:
        raise ValueError("noise_sigma must be positive")
    if minimum_count < 2:
        raise ValueError("minimum_count must be at least 2")
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same number of elements")
    if prediction.numel() == 0:
        return prediction.new_tensor(0.0), 0
    if prediction.numel() < minimum_count:
        return nn.functional.mse_loss(prediction, target), 0
    variance = prediction.new_tensor(float(noise_sigma) ** 2)
    logits = -(prediction[:, None] - target[None, :]).square() / (2.0 * variance)
    labels = torch.arange(prediction.numel(), device=prediction.device)
    loss = nn.functional.cross_entropy(logits, labels) * (2.0 * variance).detach()
    return loss, int(prediction.numel())


def ziln_expected_rate(
    log_mu: torch.Tensor,
    log_sigma: torch.Tensor,
    minimum_sigma: float = 1e-4,
) -> torch.Tensor:
    sigma = nn.functional.softplus(log_sigma) + float(minimum_sigma)
    return torch.exp(torch.clamp(log_mu + 0.5 * sigma.square(), max=20.0))


def compute_ziln_nll(
    activity_logit: torch.Tensor,
    log_mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    minimum_sigma: float = 1e-4,
) -> torch.Tensor:
    if activity_logit.shape != target.shape or log_mu.shape != target.shape or log_sigma.shape != target.shape:
        raise ValueError("ZILN parameters and target must have identical shapes")
    active_target = (target > 0.0).to(target.dtype)
    gate_nll = nn.functional.binary_cross_entropy_with_logits(
        activity_logit,
        active_target,
        reduction="none",
    )
    sigma = nn.functional.softplus(log_sigma) + float(minimum_sigma)
    safe_target = target.clamp_min(torch.finfo(target.dtype).tiny)
    lognormal_nll = (
        torch.log(safe_target)
        + torch.log(sigma)
        + 0.5 * np.log(2.0 * np.pi)
        + (torch.log(safe_target) - log_mu).square() / (2.0 * sigma.square())
    )
    return (gate_nll + active_target * lognormal_nll).mean()


def compute_active_mass_total_loss(
    predicted_mass_total: torch.Tensor,
    target: dict[str, torch.Tensor],
    active_mask: torch.Tensor,
    target_mode: str = "normalized",
    raw_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> torch.Tensor:
    if predicted_mass_total.ndim != target["link_rate"].ndim:
        raise ValueError("link_active_mass_total must have the same rank as link_rate")
    if predicted_mass_total.shape[-2] != 1:
        raise ValueError("link_active_mass_total edge dimension must be 1")
    if target_mode == "normalized":
        predicted_total = predicted_mass_total
        target_rate = target["link_rate"]
    elif target_mode == "raw":
        if raw_stats is None:
            raise ValueError("active_mass_target_mode='raw' requires active_mass_raw_stats")
        if "link_rate_raw" not in target:
            raise ValueError("active_mass_target_mode='raw' requires target link_rate_raw")
        predicted_total = inverse_normalize_tensor(predicted_mass_total, raw_stats)
        target_rate = target["link_rate_raw"]
    else:
        raise ValueError("active_mass_target_mode must be one of: normalized, raw")
    target_active_rate = torch.where(active_mask, target_rate, torch.zeros_like(target_rate))
    target_mass_total = target_active_rate.sum(dim=-2, keepdim=True)
    return ((predicted_total - target_mass_total) ** 2).mean()


def select_hurdle_training_rate_output(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    default_rate: torch.Tensor,
    hurdle_train_gate_mode: str,
    hurdle_train_gate_power: float = 1.0,
) -> torch.Tensor:
    if "link_positive_rate" not in outputs:
        return default_rate
    if "link_active_mass_rate" in outputs:
        return default_rate
    if hurdle_train_gate_mode == "none":
        return default_rate
    positive_rate = outputs["link_positive_rate"]
    activity_prob = torch.sigmoid(outputs["link_activity_logit"])
    if hurdle_train_gate_power != 1.0:
        activity_prob = calibrate_hurdle_gate_tensor(activity_prob, power=hurdle_train_gate_power)
    if hurdle_train_gate_mode == "predicted":
        return activity_prob * positive_rate
    if hurdle_train_gate_mode == "detach":
        return activity_prob.detach() * positive_rate
    if hurdle_train_gate_mode == "teacher_forcing":
        return target["link_activity"] * positive_rate
    raise ValueError("hurdle_train_gate_mode must be one of: none, predicted, detach, teacher_forcing")




def compute_activity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mode: str = "bce",
    pos_weight: float = 80.0,
    focal_gamma: float = 2.0,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pos_weight_tensor = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
    element_loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    if mode == "bce":
        return masked_mean(element_loss, loss_mask)
    if mode == "focal":
        prob = torch.sigmoid(logits)
        p_t = torch.where(target > 0.5, prob, 1.0 - prob)
        focal_weight = (1.0 - p_t).clamp(min=0.0) ** float(focal_gamma)
        return masked_mean(element_loss * focal_weight, loss_mask)
    raise ValueError("activity_loss_mode must be one of: bce, focal")


def compute_dynamic_hard_negative_activity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 0.1,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("dynamic_hard_negative_ratio must be in (0, 1]")
    inactive_mask = target <= 0.5
    if loss_mask is not None:
        inactive_mask = inactive_mask & loss_mask
    if not inactive_mask.any():
        return logits.new_tensor(0.0), 0
    inactive_logits = logits[inactive_mask]
    count = max(1, int(torch.ceil(torch.tensor(float(inactive_logits.numel()) * float(ratio))).item()))
    count = min(count, inactive_logits.numel())
    hard_logits = torch.topk(inactive_logits, k=count).values
    hard_target = torch.zeros_like(hard_logits)
    loss = nn.functional.binary_cross_entropy_with_logits(hard_logits, hard_target, reduction="mean")
    return loss, int(count)


def compute_tweedie_deviance_loss(pred: torch.Tensor, target: torch.Tensor, power: float = 1.5) -> torch.Tensor:
    if power <= 1.0 or power >= 2.0:
        raise ValueError("positive_rate_tweedie_power must be in (1, 2)")
    mu = pred.clamp_min(1e-6)
    y = target.clamp_min(0.0)
    p = float(power)
    return (-y * mu.pow(1.0 - p) / (1.0 - p) + mu.pow(2.0 - p) / (2.0 - p)).mean()


def inverse_normalize_tensor(values: torch.Tensor, stats: tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    mean, std = stats
    mean_t = torch.as_tensor(mean, device=values.device, dtype=values.dtype)
    std_t = torch.as_tensor(std, device=values.device, dtype=values.dtype)
    return values * std_t + mean_t


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if not mask.any():
        return values.new_tensor(0.0)
    return values[mask].mean()


def expand_candidate_loss_mask(
    candidate_loss_mask: torch.Tensor | np.ndarray | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if candidate_loss_mask is None:
        return None
    mask = torch.as_tensor(candidate_loss_mask, device=reference.device, dtype=torch.bool)
    if mask.ndim != 1:
        raise ValueError("candidate_loss_mask must be a one-dimensional edge mask")
    edge_count = reference.shape[-2] if reference.ndim >= 4 and reference.shape[-1] == 1 else reference.shape[-1]
    if int(mask.shape[0]) != int(edge_count):
        raise ValueError("candidate_loss_mask length must match the number of candidate edges")
    shape = [1] * reference.ndim
    edge_axis = reference.ndim - 2 if reference.ndim >= 4 and reference.shape[-1] == 1 else reference.ndim - 1
    shape[edge_axis] = int(mask.shape[0])
    return mask.reshape(shape).expand_as(reference)


def candidate_mask_edge_count(
    candidate_loss_mask: torch.Tensor | np.ndarray | None,
    reference: torch.Tensor,
) -> int:
    if candidate_loss_mask is None:
        edge_count = reference.shape[-2] if reference.ndim >= 4 and reference.shape[-1] == 1 else reference.shape[-1]
        return int(edge_count)
    return int(torch.as_tensor(candidate_loss_mask, dtype=torch.bool).sum().item())


def apply_candidate_eval_mask(
    predictions: dict[str, np.ndarray],
    candidate_eval_mask: np.ndarray | torch.Tensor,
) -> dict[str, np.ndarray]:
    mask = np.asarray(candidate_eval_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("candidate_eval_mask must be a one-dimensional edge mask")
    masked = dict(predictions)
    for name in (
        "link_activity_prob",
        "link_activity_true",
        "link_rate_pred",
        "link_positive_rate_pred",
        "link_active_rate_aux_pred",
        "link_active_mass_rate_pred",
        "link_rate_true",
    ):
        if name in masked:
            masked[name] = select_candidate_edges(masked[name], mask)
    return masked


def select_candidate_edges(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edge_axis = values.ndim - 2 if values.ndim >= 4 and values.shape[-1] == 1 else values.ndim - 1
    if values.shape[edge_axis] != mask.shape[0]:
        raise ValueError("candidate_eval_mask length must match prediction edge dimension")
    return np.take(values, np.where(mask)[0], axis=edge_axis)


def sample_inactive_loss_mask(inactive_mask: torch.Tensor, sample_ratio: float, seed: int | None = None) -> torch.Tensor:
    if sample_ratio >= 1.0:
        return inactive_mask
    sampled = torch.zeros_like(inactive_mask, dtype=torch.bool)
    inactive_indices = inactive_mask.nonzero(as_tuple=False)
    if inactive_indices.numel() == 0:
        return sampled
    sample_count = max(1, int(round(float(inactive_indices.shape[0]) * float(sample_ratio))))
    sample_count = min(sample_count, int(inactive_indices.shape[0]))
    generator = None
    if seed is not None:
        generator = torch.Generator(device=inactive_indices.device)
        generator.manual_seed(int(seed))
    order = torch.randperm(inactive_indices.shape[0], device=inactive_indices.device, generator=generator)
    chosen = inactive_indices[order[:sample_count]]
    sampled[tuple(chosen.T)] = True
    return sampled

def select_v8_link_rate_output(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor] | None = None,
    rate_output_mode: str = "main",
    inactive_rate_value: float = 0.0,
) -> torch.Tensor:
    if rate_output_mode == "main":
        return outputs["link_rate"]
    if rate_output_mode == "hurdle_gate":
        if "link_hurdle_rate" in outputs:
            return outputs["link_hurdle_rate"]
        if "link_positive_rate" not in outputs:
            raise ValueError("hurdle_gate requires link_positive_rate outputs")
        activity_prob = torch.sigmoid(outputs["link_activity_logit"])
        return activity_prob * outputs["link_positive_rate"]
    if rate_output_mode == "dual_soft_blend":
        if "link_hurdle_rate" not in outputs:
            raise ValueError("dual_soft_blend requires model_rate_output_mode='hurdle_dual'")
        activity_prob = torch.sigmoid(outputs["link_activity_logit"])
        return (1.0 - activity_prob) * outputs["link_rate"] + activity_prob * outputs["link_hurdle_rate"]
    if rate_output_mode == "active_mass_alloc":
        if "link_active_mass_rate" not in outputs:
            raise ValueError("active_mass_alloc requires model_rate_output_mode='hurdle_mass'")
        return outputs["link_active_mass_rate"]
    if "link_active_rate_aux" not in outputs:
        raise ValueError(f"{rate_output_mode} requires active_rate_auxiliary=True")
    inactive = torch.full_like(outputs["link_active_rate_aux"], float(inactive_rate_value))
    if rate_output_mode == "aux_soft_zero":
        activity_prob = torch.sigmoid(outputs["link_activity_logit"])
        return activity_prob * outputs["link_active_rate_aux"] + (1.0 - activity_prob) * inactive
    if rate_output_mode == "aux_oracle_zero":
        if target is None:
            raise ValueError("aux_oracle_zero requires target link_activity")
        return torch.where(target["link_activity"] > 0.5, outputs["link_active_rate_aux"], inactive)
    raise ValueError("rate_output_mode must be one of: main, hurdle_gate, dual_soft_blend, active_mass_alloc, aux_soft_zero, aux_oracle_zero")


def denormalize_v8_link_rate_prediction(
    values: np.ndarray,
    stats: dict,
    baseline: np.ndarray | None = None,
    clip_prediction: bool = True,
) -> np.ndarray:
    rate_space = inverse_normalize(values, stats["y_link_rate"])
    transform = stats.get("rate_target_transform", "raw")
    if transform == "residual_last_rate":
        if baseline is None:
            raise ValueError("baseline is required for residual_last_rate predictions")
        return (rate_space + baseline).astype(np.float32)
    return inverse_transform_link_rate(
        rate_space,
        transform,
        clip_max=stats.get("rate_inverse_clip_max") if clip_prediction else None,
    )


def choose_activity_threshold(prob: np.ndarray, true: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = activity_metrics(prob, true, threshold=float(threshold))["f1"]
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def mean_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    result = {}
    for key, first_value in rows[0].items():
        if isinstance(first_value, str):
            result[key] = first_value
        else:
            result[key] = float(sum(float(row[key]) for row in rows) / len(rows))
    return result
