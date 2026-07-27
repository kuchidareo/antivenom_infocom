"""Deterministic, globally balanced campaign manifest generation."""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .schema import (
    CampaignValidationError,
    SCHEMA_VERSION,
    make_context_id,
    require_identifier,
    require_positive_int,
    slug,
    validate_context,
    validate_manifest,
)


def _ready_entries(values: Any, axis: str) -> List[Mapping[str, Any]]:
    if not isinstance(values, list) or not values:
        raise CampaignValidationError("Axis '{}' must be a non-empty list".format(axis))
    result = []
    for value in values:
        if isinstance(value, str):
            entry: Mapping[str, Any] = {"id": value, "status": "ready"}
        elif isinstance(value, Mapping):
            entry = value
        else:
            raise CampaignValidationError("Invalid entry in axis '{}'".format(axis))
        if entry.get("status", "ready") == "ready":
            if "id" not in entry:
                raise CampaignValidationError("Axis '{}' entry is missing id".format(axis))
            result.append(entry)
    if not result:
        raise CampaignValidationError("Axis '{}' has no ready values".format(axis))
    return result


def _is_compatible(dataset: Mapping[str, Any], model: Mapping[str, Any]) -> bool:
    allowed_models = dataset.get("allowed_models")
    if allowed_models is not None and model["id"] not in allowed_models:
        return False
    allowed_datasets = model.get("allowed_datasets")
    if allowed_datasets is not None and dataset["id"] not in allowed_datasets:
        return False
    return True


def build_candidate_contexts(space: Mapping[str, Any]) -> List[Dict[str, Any]]:
    axes = space.get("axes")
    if not isinstance(axes, Mapping):
        raise CampaignValidationError("context space is missing axes")

    datasets = _ready_entries(axes.get("datasets"), "datasets")
    models = _ready_entries(axes.get("models"), "models")
    batch_sizes = _ready_entries(axes.get("batch_sizes"), "batch_sizes")
    backgrounds = _ready_entries(
        axes.get("background_workloads"), "background_workloads"
    )
    partitions = _ready_entries(axes.get("partition_methods"), "partition_methods")

    local_epochs = require_positive_int("local_epochs", space.get("local_epochs"))
    analysis_trials = require_positive_int(
        "analysis_trials", space.get("analysis_trials")
    )
    seed = require_positive_int("campaign_seed", space.get("campaign_seed"), True)

    contexts: List[Dict[str, Any]] = []
    for dataset, model, batch, background, partition in itertools.product(
        datasets, models, batch_sizes, backgrounds, partitions
    ):
        if not _is_compatible(dataset, model):
            continue
        supported_partitions = dataset.get("partition_methods")
        if supported_partitions and partition["id"] not in supported_partitions:
            continue

        input_size = int(dataset.get("input_size", space.get("default_input_size", 224)))
        context: Dict[str, Any] = {
            "dataset": str(dataset["id"]),
            "model": str(model["id"]),
            "input_size": input_size,
            "batch_size": int(batch["id"]),
            "background_workload": str(background["id"]),
            "partition_method": str(partition["id"]),
            "local_epochs": local_epochs,
            "reference_trials": 0,
            "analysis_trials": analysis_trials,
            "seed": seed,
            "pruning": {"method": "none", "ratio": 0.0, "structural": False},
            "memory_format": "contiguous",
            "mkldnn_enabled": True,
        }
        context["context_id"] = make_context_id(context)
        context["stage_name"] = "{}_{}".format(
            slug(dataset["id"]), context["context_id"][-8:]
        )
        validate_context(context)
        contexts.append(context)

    if not contexts:
        raise CampaignValidationError("No compatible ready contexts were generated")
    return contexts


def _candidate_score(
    context: Mapping[str, Any],
    counts: Counter,
    pair_counts: Counter,
    reuse_counts: Counter,
    jitter: Mapping[str, float],
) -> Tuple[float, str]:
    axes = (
        "dataset",
        "model",
        "batch_size",
        "background_workload",
        "partition_method",
    )
    marginal = sum(float(counts[(axis, str(context[axis]))]) for axis in axes)
    pair = float(pair_counts[(str(context["dataset"]), str(context["model"]))])
    reuse = float(reuse_counts[str(context["context_id"])])
    score = marginal + (0.75 * pair) + (0.25 * reuse) + jitter[context["context_id"]]
    return score, str(context["context_id"])


def _calibration_context(space: Mapping[str, Any]) -> Dict[str, Any]:
    calibration = space.get("calibration")
    if not isinstance(calibration, Mapping):
        raise CampaignValidationError("context space is missing calibration")
    context: Dict[str, Any] = {
        "dataset": str(calibration["dataset"]),
        "model": str(calibration["model"]),
        "input_size": int(calibration.get("input_size", 224)),
        "batch_size": int(calibration.get("batch_size", 16)),
        "background_workload": "none",
        "partition_method": str(calibration.get("partition_method", "iid")),
        "local_epochs": int(calibration.get("local_epochs", 15)),
        "reference_trials": int(calibration.get("reference_trials", 0)),
        "analysis_trials": int(calibration.get("trials", 3)),
        "seed": int(calibration.get("seed", space["campaign_seed"])),
        "pruning": {"method": "none", "ratio": 0.0, "structural": False},
        "memory_format": "contiguous",
        "mkldnn_enabled": True,
    }
    context["context_id"] = make_context_id(context)
    context["stage_name"] = "calibration_{}".format(context["context_id"][-8:])
    validate_context(context)
    return context


def load_targets(path: Path | str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    targets = document.get("targets") if isinstance(document, Mapping) else None
    if not isinstance(targets, list) or not targets:
        raise CampaignValidationError("hardware targets must contain a non-empty targets list")

    result: List[Dict[str, str]] = []
    seen = set()
    for target in targets:
        if not isinstance(target, Mapping):
            raise CampaignValidationError("Each hardware target must be an object")
        cluster = require_identifier("cluster", target.get("cluster"))
        hardware_type = require_identifier("hardware_type", target.get("hardware_type"))
        node_id = str(target.get("node_id", ""))
        if node_id:
            require_identifier("node_id", node_id)
        key = (cluster, hardware_type, node_id)
        if key in seen:
            raise CampaignValidationError("Duplicate target: {}".format(key))
        seen.add(key)
        result.append(
            {"cluster": cluster, "hardware_type": hardware_type, "node_id": node_id}
        )
    return result


def generate_manifest(
    space: Mapping[str, Any], targets: Sequence[Mapping[str, str]]
) -> Dict[str, Any]:
    campaign_id = require_identifier("campaign_id", space.get("campaign_id"))
    campaign_seed = require_positive_int(
        "campaign_seed", space.get("campaign_seed"), allow_zero=True
    )
    contexts_per_hardware = require_positive_int(
        "contexts_per_hardware", space.get("contexts_per_hardware")
    )
    candidates = build_candidate_contexts(space)
    if len(candidates) < contexts_per_hardware:
        raise CampaignValidationError(
            "Only {} ready contexts exist, but {} are required per hardware".format(
                len(candidates), contexts_per_hardware
            )
        )

    rng = random.Random(campaign_seed)
    jitter = {context["context_id"]: rng.random() * 0.01 for context in candidates}
    target_order = list(targets)
    rng.shuffle(target_order)

    counts: Counter = Counter()
    pair_counts: Counter = Counter()
    reuse_counts: Counter = Counter()
    assignments: List[Dict[str, Any]] = []

    for target in target_order:
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        for _ in range(contexts_per_hardware):
            available = [
                context
                for context in candidates
                if context["context_id"] not in selected_ids
            ]
            chosen = min(
                available,
                key=lambda context: _candidate_score(
                    context, counts, pair_counts, reuse_counts, jitter
                ),
            )
            selected.append(deepcopy(chosen))
            selected_ids.add(chosen["context_id"])
            for axis in (
                "dataset",
                "model",
                "batch_size",
                "background_workload",
                "partition_method",
            ):
                counts[(axis, str(chosen[axis]))] += 1
            pair_counts[(str(chosen["dataset"]), str(chosen["model"]))] += 1
            reuse_counts[str(chosen["context_id"])] += 1

        assignments.append(
            {
                "cluster": target["cluster"],
                "hardware_type": target["hardware_type"],
                "node_id": target.get("node_id", ""),
                "contexts": selected,
            }
        )

    assignments.sort(
        key=lambda value: (
            value["cluster"], value["hardware_type"], value.get("node_id", "")
        )
    )
    calibration = space["calibration"]
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_seed": campaign_seed,
        "contexts_per_hardware": contexts_per_hardware,
        "conditions": list(space["conditions"]),
        "calibration": {
            "enabled": bool(calibration.get("enabled", True)),
            "conditions": list(calibration.get("conditions", ["clean"])),
            "context": _calibration_context(space),
        },
        "assignments": assignments,
    }
    validate_manifest(manifest, contexts_per_hardware)
    return manifest


def plan_batches(
    targets: Sequence[Mapping[str, str]], max_batch_size: int = 10
) -> Dict[str, Any]:
    require_positive_int("max_batch_size", max_batch_size)
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for target in targets:
        grouped.setdefault(target["cluster"], []).append(target)

    batches = []
    for cluster in sorted(grouped):
        cluster_targets = sorted(
            grouped[cluster],
            key=lambda value: (value["hardware_type"], value.get("node_id", "")),
        )
        for start in range(0, len(cluster_targets), max_batch_size):
            members = cluster_targets[start : start + max_batch_size]
            batches.append(
                {
                    "batch_id": "{}-{:02d}".format(
                        cluster, (start // max_batch_size) + 1
                    ),
                    "cluster": cluster,
                    "hardware_types": [value["hardware_type"] for value in members],
                    "profile_hardware_types": ",".join(
                        value["hardware_type"] for value in members
                    ),
                }
            )
    return {"max_batch_size": max_batch_size, "batches": batches}
