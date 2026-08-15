"""Pure joint-action attempt ledger for PI-JWM P2 collection audits."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


LEDGER_SCHEMA_VERSION = "PIJWM-Action-Attempt-Ledger-v1"
RUN_ROLES = ("natural_reference", "natural_replay", "fixture", "bootstrap")
MUTATION_STATES = ("none", "confirmed", "unknown_after_runtime_call")
SETTER_KINDS = ("cpu_callback", "offload", "return_route", "rb")
SUCCESS_PATH = (
    "begun",
    "candidate_built",
    "contract_validated",
    "pre_setter_revalidated",
    "setters_applied",
    "env_step_started",
    "env_step_completed",
    "outcome_captured",
)
_REQUIRED_FIELDS = (
    "schema_version",
    "attempt_id",
    "run_role",
    "episode_id",
    "trajectory_id",
    "frame_index",
    "candidate_ordinal",
    "candidate_digest",
    "stage_trace",
    "setter_calls",
    "env_step_called",
    "env_step_completed",
    "disposition",
    "terminal_stage",
    "rejection_code",
    "rejection_detail",
    "environment_mutation_status",
    "quarantined",
    "training_eligible",
)


class LedgerContractError(ValueError):
    """An attempt identity, transition, or terminal record is invalid."""


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerContractError(f"{field} must be a non-empty string")
    return value


def _index(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerContractError(f"{field} must be a non-boolean nonnegative integer")
    return value


@dataclass(frozen=True)
class AttemptIdentity:
    run_role: str
    episode_id: str
    trajectory_id: str
    frame_index: int
    candidate_ordinal: int

    def __post_init__(self) -> None:
        if self.run_role not in RUN_ROLES:
            raise LedgerContractError(f"unknown run_role: {self.run_role!r}")
        _nonempty(self.episode_id, field="episode_id")
        _nonempty(self.trajectory_id, field="trajectory_id")
        _index(self.frame_index, field="frame_index")
        _index(self.candidate_ordinal, field="candidate_ordinal")
        if self.run_role == "natural_reference" and self.candidate_ordinal != 0:
            raise LedgerContractError(
                "natural_reference candidate_ordinal must be zero in the current protocol"
            )

    def payload(self) -> dict[str, object]:
        return {
            "run_role": self.run_role,
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "frame_index": self.frame_index,
            "candidate_ordinal": self.candidate_ordinal,
        }


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LedgerContractError("canonical mappings require string keys")
            converted[key] = _plain(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _plain(value.item())
    raise LedgerContractError(f"candidate contains unsupported value: {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            _plain(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise LedgerContractError(f"value is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def candidate_digest(value: object) -> str:
    if value is None:
        raise LedgerContractError("a built candidate cannot be null")
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _attempt_id(identity: AttemptIdentity) -> str:
    return f"attempt::{hashlib.sha256(_canonical_bytes(identity.payload())).hexdigest()}"


class ActionAttemptLedger:
    """Own active handles and emit each terminal attempt exactly once."""

    def __init__(self) -> None:
        self._handles: dict[str, AttemptHandle] = {}
        self._terminal_rows: list[dict[str, object]] = []

    def begin_id(self, identity: AttemptIdentity) -> str:
        if not isinstance(identity, AttemptIdentity):
            raise LedgerContractError("identity must be AttemptIdentity")
        return _attempt_id(identity)

    def begin(self, identity: AttemptIdentity) -> "AttemptHandle":
        attempt_id = self.begin_id(identity)
        if attempt_id in self._handles:
            raise LedgerContractError(f"duplicate attempt_id: {attempt_id}")
        handle = AttemptHandle(self, identity, attempt_id)
        self._handles[attempt_id] = handle
        return handle

    def _commit(self, handle: "AttemptHandle", row: dict[str, object]) -> None:
        if handle._terminal:
            raise LedgerContractError(f"attempt is already terminal: {handle.attempt_id}")
        if any(existing["attempt_id"] == handle.attempt_id for existing in self._terminal_rows):
            raise LedgerContractError(f"duplicate terminal attempt_id: {handle.attempt_id}")
        errors = validate_attempt_records([row])
        if errors:
            raise LedgerContractError(f"terminal record invalid: {list(errors)}")
        handle._terminal = True
        self._terminal_rows.append(copy.deepcopy(row))

    def terminal_records(self) -> list[dict[str, object]]:
        return copy.deepcopy(self._terminal_rows)


class AttemptHandle:
    def __init__(
        self,
        ledger: ActionAttemptLedger,
        identity: AttemptIdentity,
        attempt_id: str,
    ) -> None:
        self._ledger = ledger
        self.identity = identity
        self.attempt_id = attempt_id
        self._stage_trace = ["begun"]
        self._candidate_digest: str | None = None
        self._setter_calls: list[dict[str, object]] = []
        self._env_step_called = False
        self._env_step_completed = False
        self._terminal = False

    @property
    def terminal(self) -> bool:
        return self._terminal

    def _require_active(self) -> None:
        if self._terminal:
            raise LedgerContractError(f"attempt is already terminal: {self.attempt_id}")

    def _advance(self, expected: str, target: str) -> None:
        self._require_active()
        observed = self._stage_trace[-1]
        if observed != expected:
            raise LedgerContractError(
                f"illegal transition from {observed!r} to {target!r}; expected {expected!r}"
            )
        self._stage_trace.append(target)

    def candidate_built(self, candidate: object) -> None:
        self._advance("begun", "candidate_built")
        self._candidate_digest = candidate_digest(candidate)

    def contract_validated(self) -> None:
        self._advance("candidate_built", "contract_validated")

    def pre_setter_revalidated(self) -> None:
        self._advance("contract_validated", "pre_setter_revalidated")

    def setter_started(self, *, setter_kind: str, task_id: str | None) -> int:
        self._require_active()
        if self._stage_trace[-1] != "pre_setter_revalidated":
            raise LedgerContractError("setter call requires pre_setter_revalidated stage")
        if setter_kind not in SETTER_KINDS:
            raise LedgerContractError(f"unknown setter_kind: {setter_kind!r}")
        if task_id is not None:
            _nonempty(task_id, field="task_id")
        if self._setter_calls and self._setter_calls[-1]["call_started"] is True \
                and self._setter_calls[-1]["call_completed"] is None:
            raise LedgerContractError("previous setter call has not finished")
        ordinal = len(self._setter_calls)
        self._setter_calls.append(
            {
                "ordinal": ordinal,
                "setter_kind": setter_kind,
                "task_id": task_id,
                "call_started": True,
                "call_completed": None,
                "succeeded": None,
                "error_type": None,
                "error_detail": None,
            }
        )
        return ordinal

    def setter_finished(
        self,
        ordinal: int,
        *,
        call_completed: bool,
        succeeded: bool,
        error_type: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        self._require_active()
        _index(ordinal, field="setter ordinal")
        if ordinal >= len(self._setter_calls):
            raise LedgerContractError(f"unknown setter ordinal: {ordinal}")
        row = self._setter_calls[ordinal]
        if row["call_completed"] is not None:
            raise LedgerContractError(f"setter call is already finished: {ordinal}")
        if not isinstance(call_completed, bool) or not isinstance(succeeded, bool):
            raise LedgerContractError("setter completion flags must be boolean")
        if succeeded and not call_completed:
            raise LedgerContractError("a succeeded setter must have completed")
        if succeeded:
            if error_type is not None or error_detail is not None:
                raise LedgerContractError("a succeeded setter cannot carry an error")
        else:
            _nonempty(error_type, field="setter error_type")
            _nonempty(error_detail, field="setter error_detail")
        row.update(
            {
                "call_completed": call_completed,
                "succeeded": succeeded,
                "error_type": error_type,
                "error_detail": error_detail,
            }
        )

    def setters_applied(self) -> None:
        if any(call["call_completed"] is None for call in self._setter_calls):
            raise LedgerContractError("cannot finish setter stage with an open call")
        self._advance("pre_setter_revalidated", "setters_applied")

    def env_step_started(self) -> None:
        self._advance("setters_applied", "env_step_started")
        self._env_step_called = True

    def env_step_completed(self) -> None:
        self._advance("env_step_started", "env_step_completed")
        self._env_step_completed = True

    def outcome_captured(self) -> None:
        self._advance("env_step_completed", "outcome_captured")

    def _row(
        self,
        *,
        disposition: str,
        terminal_stage: str,
        rejection_code: str | None,
        rejection_detail: str | None,
        environment_mutation_status: str,
        quarantined: bool,
    ) -> dict[str, object]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            **self.identity.payload(),
            "candidate_digest": self._candidate_digest,
            "stage_trace": list(self._stage_trace),
            "setter_calls": copy.deepcopy(self._setter_calls),
            "env_step_called": self._env_step_called,
            "env_step_completed": self._env_step_completed,
            "disposition": disposition,
            "terminal_stage": terminal_stage,
            "rejection_code": rejection_code,
            "rejection_detail": rejection_detail,
            "environment_mutation_status": environment_mutation_status,
            "quarantined": quarantined,
            "training_eligible": False,
        }

    def accept(self) -> dict[str, object]:
        self._require_active()
        if self._stage_trace[-1] != "outcome_captured":
            raise LedgerContractError("accepted attempt requires outcome_captured")
        row = self._row(
            disposition="accepted",
            terminal_stage="outcome_captured",
            rejection_code=None,
            rejection_detail=None,
            environment_mutation_status="confirmed",
            quarantined=False,
        )
        self._ledger._commit(self, row)
        return copy.deepcopy(row)

    def reject(
        self,
        *,
        terminal_stage: str,
        rejection_code: str,
        rejection_detail: str,
        environment_mutation_status: str,
    ) -> dict[str, object]:
        self._require_active()
        _nonempty(terminal_stage, field="terminal_stage")
        _nonempty(rejection_code, field="rejection_code")
        _nonempty(rejection_detail, field="rejection_detail")
        if environment_mutation_status not in MUTATION_STATES:
            raise LedgerContractError(
                f"unknown environment_mutation_status: {environment_mutation_status!r}"
            )
        runtime_started = bool(self._setter_calls) or self._env_step_called
        successful_setter = any(call["succeeded"] is True for call in self._setter_calls)
        if environment_mutation_status == "none" and runtime_started:
            raise LedgerContractError("mutation none contradicts a started runtime action call")
        if environment_mutation_status == "unknown_after_runtime_call":
            if not self._setter_calls or successful_setter or self._env_step_called:
                raise LedgerContractError("unknown mutation requires only failed setter runtime evidence")
        if environment_mutation_status == "confirmed" and not (
            successful_setter or self._env_step_called
        ):
            raise LedgerContractError("confirmed mutation lacks a successful setter or env.step call")
        quarantined = environment_mutation_status != "none"
        row = self._row(
            disposition="rejected",
            terminal_stage=terminal_stage,
            rejection_code=rejection_code,
            rejection_detail=rejection_detail,
            environment_mutation_status=environment_mutation_status,
            quarantined=quarantined,
        )
        self._ledger._commit(self, row)
        return copy.deepcopy(row)


def _record_errors(row: Mapping[str, object], *, index: int) -> list[str]:
    errors: list[str] = []
    prefix = f"attempt[{index}]"
    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        return [f"{prefix} missing fields: {missing}"]
    if row.get("schema_version") != LEDGER_SCHEMA_VERSION:
        errors.append(f"{prefix} schema_version mismatch")
    try:
        identity = AttemptIdentity(
            run_role=row.get("run_role"),  # type: ignore[arg-type]
            episode_id=row.get("episode_id"),  # type: ignore[arg-type]
            trajectory_id=row.get("trajectory_id"),  # type: ignore[arg-type]
            frame_index=row.get("frame_index"),  # type: ignore[arg-type]
            candidate_ordinal=row.get("candidate_ordinal"),  # type: ignore[arg-type]
        )
    except LedgerContractError as exc:
        errors.append(f"{prefix} identity invalid: {exc}")
        identity = None
    if identity is not None and row.get("attempt_id") != _attempt_id(identity):
        errors.append(f"{prefix} attempt_id mismatch")

    digest = row.get("candidate_digest")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        errors.append(f"{prefix} candidate_digest is not lowercase SHA-256")
    trace = row.get("stage_trace")
    if not isinstance(trace, list) or not trace or trace != list(SUCCESS_PATH[: len(trace)]):
        errors.append(f"{prefix} stage_trace is not a contiguous success-path prefix")

    setters = row.get("setter_calls")
    if not isinstance(setters, list):
        errors.append(f"{prefix} setter_calls must be an array")
        setters = []
    for ordinal, call in enumerate(setters):
        if not isinstance(call, Mapping):
            errors.append(f"{prefix} setter_calls[{ordinal}] must be an object")
            continue
        if call.get("ordinal") != ordinal:
            errors.append(f"{prefix} setter ordinal sequence mismatch")
        if call.get("setter_kind") not in SETTER_KINDS:
            errors.append(f"{prefix} setter kind invalid")
        if call.get("call_started") is not True:
            errors.append(f"{prefix} setter call_started must be true")
        if not isinstance(call.get("call_completed"), bool):
            errors.append(f"{prefix} setter call_completed must be boolean")
        if not isinstance(call.get("succeeded"), bool):
            errors.append(f"{prefix} setter succeeded must be boolean")
        if call.get("succeeded") is True and call.get("call_completed") is not True:
            errors.append(f"{prefix} succeeded setter did not complete")
        if call.get("succeeded") is True and (
            call.get("error_type") is not None or call.get("error_detail") is not None
        ):
            errors.append(f"{prefix} succeeded setter carries an error")
        if call.get("succeeded") is False and (
            not isinstance(call.get("error_type"), str)
            or not str(call.get("error_type")).strip()
            or not isinstance(call.get("error_detail"), str)
            or not str(call.get("error_detail")).strip()
        ):
            errors.append(f"{prefix} failed setter lacks error evidence")

    step_called = row.get("env_step_called")
    step_completed = row.get("env_step_completed")
    if not isinstance(step_called, bool) or not isinstance(step_completed, bool):
        errors.append(f"{prefix} env.step flags must be boolean")
    if step_completed is True and step_called is not True:
        errors.append(f"{prefix} completed env.step was not called")
    if row.get("training_eligible") is not False:
        errors.append(f"{prefix} training_eligible must be false")
    mutation = row.get("environment_mutation_status")
    if mutation not in MUTATION_STATES:
        errors.append(f"{prefix} mutation status invalid")
    quarantined = row.get("quarantined")
    if not isinstance(quarantined, bool):
        errors.append(f"{prefix} quarantined must be boolean")

    runtime_started = bool(setters) or step_called is True
    successful_setter = any(
        isinstance(call, Mapping) and call.get("succeeded") is True for call in setters
    )
    if mutation == "none":
        if runtime_started or quarantined is not False:
            errors.append(f"{prefix} mutation none contradicts runtime/quarantine evidence")
    elif mutation == "unknown_after_runtime_call":
        if not setters or successful_setter or step_called is True or quarantined is not True:
            errors.append(f"{prefix} unknown mutation evidence is inconsistent")
    elif mutation == "confirmed":
        if not (successful_setter or step_called is True):
            errors.append(f"{prefix} confirmed mutation lacks runtime evidence")

    disposition = row.get("disposition")
    if disposition == "accepted":
        if trace != list(SUCCESS_PATH):
            errors.append(f"{prefix} accepted attempt lacks complete stage path")
        if digest is None:
            errors.append(f"{prefix} accepted attempt lacks candidate digest")
        if step_called is not True or step_completed is not True:
            errors.append(f"{prefix} accepted attempt lacks completed env.step")
        if row.get("terminal_stage") != "outcome_captured":
            errors.append(f"{prefix} accepted terminal_stage mismatch")
        if row.get("rejection_code") is not None or row.get("rejection_detail") is not None:
            errors.append(f"{prefix} accepted attempt carries rejection evidence")
        if mutation != "confirmed" or quarantined is not False:
            errors.append(f"{prefix} accepted mutation/quarantine evidence is inconsistent")
    elif disposition == "rejected":
        for field in ("terminal_stage", "rejection_code", "rejection_detail"):
            if not isinstance(row.get(field), str) or not str(row.get(field)).strip():
                errors.append(f"{prefix} rejected attempt lacks {field}")
        if mutation != "none" and quarantined is not True:
            errors.append(f"{prefix} mutated rejection must be quarantined")
    else:
        errors.append(f"{prefix} disposition must be accepted or rejected")
    return errors


def validate_attempt_records(
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        return ("attempt records must be a sequence",)
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"attempt[{index}] must be an object")
            continue
        errors.extend(_record_errors(row, index=index))
        attempt_id = row.get("attempt_id")
        if isinstance(attempt_id, str):
            if attempt_id in seen:
                errors.append(f"duplicate attempt_id: {attempt_id}")
            seen.add(attempt_id)
    return tuple(errors)


def summarize_attempts(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    errors = validate_attempt_records(rows)
    if errors:
        raise LedgerContractError(f"cannot summarize invalid attempt records: {list(errors)}")
    by_role: dict[str, dict[str, object]] = {}
    for role in RUN_ROLES:
        selected = [row for row in rows if row["run_role"] == role]
        accepted = sum(row["disposition"] == "accepted" for row in selected)
        rejected = sum(row["disposition"] == "rejected" for row in selected)
        count = len(selected)
        by_role[role] = {
            "attempt_count": count,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "rejection_rate": (rejected / count) if count else None,
            "quarantined_count": sum(row["quarantined"] is True for row in selected),
            "mutation_counts": {
                state: sum(row["environment_mutation_status"] == state for row in selected)
                for state in MUTATION_STATES
            },
            "binary_conservation_passed": count == accepted + rejected,
        }
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "by_run_role": by_role,
        "natural_reference_rejection_rate": by_role["natural_reference"]["rejection_rate"],
        "binary_conservation_passed": all(
            row["binary_conservation_passed"] is True for row in by_role.values()
        ),
    }
