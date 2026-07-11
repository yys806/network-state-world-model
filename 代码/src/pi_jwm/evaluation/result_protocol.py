"""Machine-readable boundaries for deployable and diagnostic results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


RESULT_KINDS = {
    "deployable",
    "true_future_reference",
    "sample_oracle",
    "test_best_diagnostic",
}
SPLITS = {"train", "val", "test", "none"}
LABEL_AWARE_ACTION_GENERATORS = {
    "true_future",
    "true_activity_policy_value",
    "policy_activity_true_value",
}


def validate_result_protocol(protocol: Mapping[str, object]) -> None:
    result_kind = str(protocol.get("result_kind", ""))
    if result_kind not in RESULT_KINDS:
        raise ValueError(f"Unknown result_kind: {result_kind!r}")

    fit_splits = tuple(str(split) for split in protocol.get("fit_splits", ()))
    selection_split = str(protocol.get("selection_split", ""))
    evaluation_split = str(protocol.get("evaluation_split", ""))
    unknown_splits = (set(fit_splits) | {selection_split, evaluation_split}) - SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown split names in result protocol: {sorted(unknown_splits)}")

    if result_kind == "deployable":
        if "test" in fit_splits:
            raise ValueError("A deployable result cannot use the test split for fitting.")
        if selection_split == "test":
            raise ValueError("A deployable result cannot select a configuration on the test split.")


def build_result_protocol(
    result_kind: str,
    fit_splits: Iterable[str],
    selection_split: str,
    evaluation_split: str,
) -> dict[str, object]:
    protocol: dict[str, object] = {
        "result_kind": str(result_kind),
        "fit_splits": [str(split) for split in fit_splits],
        "selection_split": str(selection_split),
        "evaluation_split": str(evaluation_split),
    }
    validate_result_protocol(protocol)
    return protocol


def classify_bridge_result(action_generator: str, action_decoder: str, mode: str) -> str:
    if (
        str(action_generator) in LABEL_AWARE_ACTION_GENERATORS
        or str(action_generator).startswith("true_")
        or str(action_decoder) == "oracle_topk"
        or str(mode) != "predicted_all"
    ):
        return "true_future_reference"
    return "deployable"
