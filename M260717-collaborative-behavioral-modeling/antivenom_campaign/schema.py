"""Schema helpers shared by campaign generation and execution.

The module intentionally uses only the Python standard library so that a
campaign can be inspected and validated before the experiment dependencies
are installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


SCHEMA_VERSION = 1
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PARTITION_METHODS = frozenset(("iid", "dirichlet_noniid"))
BACKGROUND_WORKLOADS = frozenset(("none", "group1", "group2", "both"))


class CampaignValidationError(ValueError):
    """Raised when a context space or generated manifest is invalid."""


def load_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CampaignValidationError("Expected a JSON object: {}".format(path))
    return value


def write_json_atomic(path: Path | str, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_context_id(context: Mapping[str, Any]) -> str:
    identity = {
        key: value
        for key, value in context.items()
        if key not in ("context_id", "stage_name")
    }
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return "ctx_{}".format(digest[:16])


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not normalized:
        raise CampaignValidationError("Cannot create a slug from {!r}".format(value))
    return normalized


def require_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER.fullmatch(value):
        raise CampaignValidationError(
            "{} must contain only letters, digits, '.', '_' or '-': {!r}".format(
                name, value
            )
        )
    return value


def require_positive_int(name: str, value: Any, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignValidationError("{} must be an integer".format(name))
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise CampaignValidationError("{} must be at least {}".format(name, minimum))
    return value


def validate_context(context: Mapping[str, Any]) -> None:
    if not isinstance(context, Mapping):
        raise CampaignValidationError("A context must be a JSON object")

    required_strings = ("context_id", "stage_name", "dataset", "model")
    for name in required_strings:
        value = context.get(name)
        if not isinstance(value, str) or not value:
            raise CampaignValidationError("Context field '{}' is required".format(name))

    require_identifier("context_id", context["context_id"])
    require_identifier("stage_name", context["stage_name"])
    require_positive_int("input_size", context.get("input_size"))
    require_positive_int("batch_size", context.get("batch_size"))
    require_positive_int("local_epochs", context.get("local_epochs"))
    require_positive_int(
        "reference_trials", context.get("reference_trials"), allow_zero=True
    )
    require_positive_int(
        "analysis_trials", context.get("analysis_trials"), allow_zero=True
    )
    require_positive_int("seed", context.get("seed"), allow_zero=True)

    partition_method = context.get("partition_method")
    if partition_method not in PARTITION_METHODS:
        raise CampaignValidationError(
            "Unsupported partition_method: {!r}".format(partition_method)
        )

    background = context.get("background_workload")
    if background not in BACKGROUND_WORKLOADS:
        raise CampaignValidationError(
            "Unsupported background_workload: {!r}".format(background)
        )

    pruning = context.get("pruning", {"method": "none", "ratio": 0.0})
    if not isinstance(pruning, Mapping):
        raise CampaignValidationError("pruning must be a JSON object")
    method = pruning.get("method", "none")
    ratio = pruning.get("ratio", 0.0)
    if not isinstance(method, str) or not method:
        raise CampaignValidationError("pruning.method must be a string")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
        raise CampaignValidationError("pruning.ratio must be numeric")
    if float(ratio) < 0.0 or float(ratio) >= 1.0:
        raise CampaignValidationError("pruning.ratio must be in [0, 1)")

    expected_id = make_context_id(context)
    if context["context_id"] != expected_id:
        raise CampaignValidationError(
            "Context ID {} does not match its contents (expected {})".format(
                context["context_id"], expected_id
            )
        )


def _validate_conditions(conditions: Any, name: str = "conditions") -> List[str]:
    if not isinstance(conditions, list) or not conditions:
        raise CampaignValidationError("{} must be a non-empty list".format(name))
    result: List[str] = []
    for condition in conditions:
        result.append(require_identifier(name, condition))
    if len(set(result)) != len(result):
        raise CampaignValidationError("{} contains duplicates".format(name))
    return result


def validate_manifest(
    manifest: Mapping[str, Any],
    expected_contexts_per_hardware: Optional[int] = None,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CampaignValidationError(
            "Unsupported schema_version: {!r}".format(manifest.get("schema_version"))
        )
    require_identifier("campaign_id", manifest.get("campaign_id"))
    require_positive_int("campaign_seed", manifest.get("campaign_seed"), allow_zero=True)
    contexts_per_hardware = require_positive_int(
        "contexts_per_hardware", manifest.get("contexts_per_hardware")
    )
    if (
        expected_contexts_per_hardware is not None
        and contexts_per_hardware != expected_contexts_per_hardware
    ):
        raise CampaignValidationError(
            "Expected {} contexts per hardware, found {}".format(
                expected_contexts_per_hardware, contexts_per_hardware
            )
        )
    _validate_conditions(manifest.get("conditions"))

    calibration = manifest.get("calibration")
    if not isinstance(calibration, Mapping):
        raise CampaignValidationError("calibration must be a JSON object")
    if calibration.get("enabled", True):
        validate_context(calibration.get("context", {}))
        _validate_conditions(calibration.get("conditions"), "calibration.conditions")

    assignments = manifest.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise CampaignValidationError("assignments must be a non-empty list")

    assignment_keys = set()
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise CampaignValidationError("Each assignment must be a JSON object")
        cluster = require_identifier("cluster", assignment.get("cluster"))
        hardware_type = require_identifier(
            "hardware_type", assignment.get("hardware_type")
        )
        node_id = assignment.get("node_id", "")
        if node_id:
            require_identifier("node_id", node_id)
        key = (cluster, hardware_type, node_id)
        if key in assignment_keys:
            raise CampaignValidationError("Duplicate hardware assignment: {}".format(key))
        assignment_keys.add(key)

        contexts = assignment.get("contexts")
        if not isinstance(contexts, list):
            raise CampaignValidationError("assignment.contexts must be a list")
        if len(contexts) != contexts_per_hardware:
            raise CampaignValidationError(
                "{}:{} has {} contexts; expected {}".format(
                    cluster, hardware_type, len(contexts), contexts_per_hardware
                )
            )
        context_ids = []
        for context in contexts:
            validate_context(context)
            context_ids.append(context["context_id"])
        if len(set(context_ids)) != len(context_ids):
            raise CampaignValidationError(
                "{}:{} contains duplicate contexts".format(cluster, hardware_type)
            )


def select_assignment(
    manifest: Mapping[str, Any],
    cluster: str,
    hardware_type: str,
    node_id: str = "",
) -> Mapping[str, Any]:
    candidates = [
        assignment
        for assignment in manifest["assignments"]
        if assignment["cluster"] == cluster
        and assignment["hardware_type"] == hardware_type
    ]
    if node_id:
        exact = [item for item in candidates if item.get("node_id", "") == node_id]
        if exact:
            candidates = exact
    if not candidates:
        raise CampaignValidationError(
            "No assignment for cluster={} hardware_type={} node_id={}".format(
                cluster, hardware_type, node_id or "<unspecified>"
            )
        )
    if len(candidates) > 1:
        raise CampaignValidationError(
            "Multiple assignments match {}:{}; provide node_id".format(
                cluster, hardware_type
            )
        )
    return candidates[0]


def manifest_datasets(
    manifest: Mapping[str, Any],
    cluster: Optional[str] = None,
    hardware_type: Optional[str] = None,
    node_id: str = "",
) -> List[str]:
    contexts: List[Mapping[str, Any]] = []
    calibration = manifest.get("calibration", {})
    if calibration.get("enabled", True):
        contexts.append(calibration["context"])

    if cluster is not None and hardware_type is not None:
        assignments: Iterable[Mapping[str, Any]] = (
            select_assignment(manifest, cluster, hardware_type, node_id),
        )
    else:
        assignments = manifest["assignments"]
    for assignment in assignments:
        contexts.extend(assignment["contexts"])
    return sorted({str(context["dataset"]) for context in contexts})
