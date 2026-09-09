"""Probability utilities for binary heads trained with positive class weights."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as functional


def _require_finite_tensor(value: torch.Tensor, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or torch.is_complex(value)
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(f"{name} must be a finite tensor")


def _binary_labels(labels: torch.Tensor, name: str = "labels") -> torch.Tensor:
    _require_finite_tensor(labels, name)
    if labels.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.logical_or(labels == 0, labels == 1).all().item():
        raise ValueError(f"{name} must be binary")
    return labels


@dataclass(frozen=True)
class InverseTemperatureCalibration:
    pos_weight: float
    log_temperature: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.pos_weight) or self.pos_weight <= 0.0:
            raise ValueError("pos_weight must be finite and positive")
        if not math.isfinite(self.log_temperature):
            raise ValueError("log_temperature must be finite")
        self.temperature

    @property
    def temperature(self) -> float:
        if abs(self.log_temperature) <= 1e-12:
            return 1.0
        try:
            temperature = math.exp(self.log_temperature)
        except OverflowError as error:
            raise ValueError("temperature must be finite and positive") from error
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        return temperature

    def unweighted_logits(self, weighted_logits: torch.Tensor) -> torch.Tensor:
        _require_finite_tensor(weighted_logits, "weighted_logits")
        return weighted_logits - math.log(self.pos_weight)

    def probabilities_from_weighted_logits(
        self, weighted_logits: torch.Tensor
    ) -> torch.Tensor:
        logits = self.unweighted_logits(weighted_logits)
        probabilities = torch.sigmoid(logits / self.temperature)
        if not torch.isfinite(probabilities).all().item():
            raise ValueError("calibrated probabilities must be finite")
        return probabilities

    def map_raw_threshold(self, raw_threshold: float) -> float:
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("raw_threshold must be finite and in (0, 1)")
        threshold_tensor = torch.tensor(threshold, dtype=torch.float64)
        mapped = torch.sigmoid(
            (torch.logit(threshold_tensor) - math.log(self.pos_weight))
            / self.temperature
        ).item()
        if not math.isfinite(mapped) or not 0.0 < mapped < 1.0:
            raise ValueError("mapped threshold must be finite and in (0, 1)")
        return mapped


def unweighted_logits(
    weighted_logits: torch.Tensor,
    calibration: InverseTemperatureCalibration,
) -> torch.Tensor:
    return calibration.unweighted_logits(weighted_logits)


def probabilities_from_weighted_logits(
    weighted_logits: torch.Tensor,
    calibration: InverseTemperatureCalibration,
) -> torch.Tensor:
    return calibration.probabilities_from_weighted_logits(weighted_logits)


def map_raw_threshold(
    raw_threshold: float,
    calibration: InverseTemperatureCalibration,
) -> float:
    return calibration.map_raw_threshold(raw_threshold)


def fit_inverse_temperature(
    weighted_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    pos_weight: float,
    fit_split: str = "calibration",
) -> tuple[InverseTemperatureCalibration, dict[str, float | str]]:
    if fit_split != "calibration":
        raise ValueError("temperature may only be fit on the calibration split")
    _require_finite_tensor(weighted_logits, "weighted_logits")
    _binary_labels(labels)
    if weighted_logits.numel() == 0 or weighted_logits.shape != labels.shape:
        raise ValueError("weighted_logits and labels must be non-empty and have equal shapes")
    base = InverseTemperatureCalibration(float(pos_weight), 0.0)
    logits = base.unweighted_logits(weighted_logits).detach().to(
        device="cpu", dtype=torch.float64
    ).reshape(-1)
    targets = labels.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    q = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device="cpu"))

    def objective() -> torch.Tensor:
        return functional.binary_cross_entropy_with_logits(logits / torch.exp(q), targets)

    identity_nll = float(objective().detach().item())
    optimizer = torch.optim.LBFGS(
        [q],
        max_iter=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    calibration = InverseTemperatureCalibration(float(pos_weight), float(q.detach().item()))
    fitted_nll = float(objective().detach().item())
    if fitted_nll > identity_nll + 1e-12:
        raise ValueError("fitted NLL is worse than identity-temperature NLL")
    return calibration, {
        "fit_split": fit_split,
        "objective": "unweighted_bernoulli_nll",
        "identity_nll": identity_nll,
        "fitted_nll": fitted_nll,
        "temperature": calibration.temperature,
        "optimizer": "LBFGS",
        "dtype": "float64",
        "device": "cpu",
        "sample_count": int(logits.numel()),
        "max_iter": 100,
        "tolerance_grad": 1e-12,
        "tolerance_change": 1e-12,
        "line_search_fn": "strong_wolfe",
    }


def binary_probability_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    ece_bin_count: int = 15,
) -> dict[str, object]:
    _require_finite_tensor(probabilities, "probabilities")
    _binary_labels(labels)
    if probabilities.numel() == 0 or probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must be non-empty and have equal shapes")
    if ece_bin_count != 15:
        raise ValueError("ece_bin_count must be 15")
    if not torch.logical_and(probabilities >= 0.0, probabilities <= 1.0).all().item():
        raise ValueError("probabilities must be in [0, 1]")
    probability_values = probabilities.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    target_values = labels.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    nll = functional.binary_cross_entropy(
        probability_values.clamp(1e-7, 1.0 - 1e-7), target_values
    ).item()
    brier = torch.mean((probability_values - target_values) ** 2).item()
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    count = int(probability_values.numel())
    for index in range(15):
        low, high = index / 15.0, (index + 1) / 15.0
        mask = (probability_values >= low) & (
            probability_values <= high if index == 14 else probability_values < high
        )
        bin_count = int(mask.sum().item())
        entry: dict[str, float | int] = {"index": index, "lower": low, "upper": high, "count": bin_count}
        if bin_count:
            mean_probability = float(probability_values[mask].mean().item())
            empirical_frequency = float(target_values[mask].mean().item())
            gap = abs(mean_probability - empirical_frequency)
            entry.update(
                mean_probability=mean_probability,
                empirical_frequency=empirical_frequency,
                gap=gap,
            )
            ece += (bin_count / count) * gap
        bins.append(entry)
    return {"count": count, "nll": nll, "brier": brier, "ece": ece, "ece_bin_count": 15, "ece_bins": bins}


def choose_legacy_threshold(
    raw_scores: torch.Tensor,
    labels: torch.Tensor,
    calibration: InverseTemperatureCalibration,
    raw_thresholds: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> dict[str, object]:
    _require_finite_tensor(raw_scores, "raw_scores")
    _binary_labels(labels)
    if raw_scores.numel() == 0 or raw_scores.shape != labels.shape:
        raise ValueError("raw_scores and labels must be non-empty and have equal shapes")
    if not torch.logical_and(raw_scores >= 0.0, raw_scores <= 1.0).all().item():
        raise ValueError("raw_scores must be in [0, 1]")
    if not raw_thresholds:
        raise ValueError("raw_thresholds must not be empty")
    raw = raw_scores.detach().to(device="cpu", dtype=torch.float64)
    target = labels.detach().to(device="cpu").bool()
    interior = (raw > 0.0) & (raw < 1.0)
    probabilities = torch.empty_like(raw)
    probabilities[raw == 0.0] = 0.0
    probabilities[raw == 1.0] = 1.0
    probabilities[interior] = torch.sigmoid(
        (torch.logit(raw[interior]) - math.log(calibration.pos_weight))
        / calibration.temperature
    )
    candidates: list[dict[str, float | int | bool]] = []
    for raw_threshold in raw_thresholds:
        mapped_threshold = map_raw_threshold(raw_threshold, calibration)
        raw_decisions = raw >= raw_threshold
        probability_decisions = probabilities >= mapped_threshold
        equivalent = bool(torch.equal(raw_decisions, probability_decisions))
        if not equivalent:
            raise ValueError("mapped probability decisions differ from legacy decisions")
        tp = int((raw_decisions & target).sum().item())
        fp = int((raw_decisions & ~target).sum().item())
        fn = int((~raw_decisions & target).sum().item())
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else (2.0 * tp) / denominator
        candidates.append({
            "raw_threshold": float(raw_threshold),
            "probability_threshold": mapped_threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": f1,
            "decision_equivalent": equivalent,
        })
    selected = max(candidates, key=lambda item: (float(item["f1"]), -abs(float(item["raw_threshold"]) - 0.5)))
    return {"selected": selected, "candidates": candidates, "decision_equivalence_to_legacy": True}


def correct_positive_weighted_probability(
    weighted_probability: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """Recover the unweighted posterior implied by weighted BCE scores."""

    weight = float(pos_weight)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("pos_weight must be finite and positive")
    score = weighted_probability.clamp(0.0, 1.0)
    return score / (weight - (weight - 1.0) * score)
