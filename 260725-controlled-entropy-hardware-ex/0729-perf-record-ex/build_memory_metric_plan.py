#!/usr/bin/env python3
"""Expand perf memory metric presets into repeatable raw-event passes."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path


TARGET_METRICS = {
    "CacheMisses": (
        "tma_l3_bound",
        "tma_l2_bound",
        "tma_l1_bound",
        "tma_info_memory_l2mpki_all",
        "tma_info_memory_l2hpki_all",
        "tma_info_memory_l3mpki",
        "tma_info_memory_l2mpki_load",
        "tma_info_memory_l2mpki",
        "tma_info_memory_l2hpki_load",
        "tma_info_memory_l1mpki_load",
        "tma_info_memory_l1mpki",
        "tma_info_memory_fb_hpki",
    ),
    "MemoryBound": (
        "tma_l3_bound",
        "tma_l2_bound",
        "tma_dram_bound",
        "tma_store_bound",
        "tma_l1_bound",
        "tma_info_memory_load_miss_real_latency",
        "tma_info_memory_mlp",
    ),
    "MemoryLat": (
        "tma_info_bottleneck_memory_latency",
        "tma_mem_latency",
        "tma_l3_hit_latency",
        "tma_store_latency",
        "tma_info_system_mem_read_latency",
        "tma_info_memory_load_miss_real_latency",
    ),
    "MemoryBW": (
        "tma_info_bottleneck_memory_bandwidth",
        "tma_mem_bandwidth",
        "tma_sq_full",
        "tma_streaming_stores",
        "tma_info_system_mem_parallel_reads",
        "tma_info_system_dram_bw_use",
        "tma_info_memory_mlp",
        "tma_fb_full",
        "tma_info_memory_core_l3_cache_fill_bw",
        "tma_info_memory_thread_l3_cache_fill_bw_1t",
        "tma_info_memory_core_l3_cache_access_bw",
        "tma_info_memory_thread_l3_cache_access_bw_1t",
        "tma_info_memory_core_l2_cache_fill_bw",
        "tma_info_memory_thread_l2_cache_fill_bw_1t",
        "tma_info_memory_core_l1d_cache_fill_bw",
        "tma_info_memory_thread_l1d_cache_fill_bw_1t",
    ),
    "MemoryTLB": (
        "tma_info_bottleneck_memory_data_tlbs",
        "tma_info_bottleneck_big_code",
        "tma_dtlb_load",
        "tma_load_stlb_hit",
        "tma_load_stlb_miss",
        "tma_dtlb_store",
        "tma_store_stlb_hit",
        "tma_store_stlb_miss",
        "tma_itlb_misses",
        "tma_info_memory_tlb_page_walks_utilization",
        "tma_info_memory_tlb_store_stlb_mpki",
        "tma_info_memory_tlb_load_stlb_mpki",
        "tma_info_memory_tlb_code_stlb_mpki",
    ),
}

BASIC_EVENTS = (
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
)

BASIC_METRICS = {
    "basic_ipc": "instructions / cycles",
    "basic_branch_miss_percent": "100 * branch-misses / branches",
    "basic_l1d_load_miss_percent": "100 * L1-dcache-load-misses / L1-dcache-loads",
}

SYSTEM_METRICS = {
    "tma_info_system_mem_read_latency",
    "tma_info_system_mem_parallel_reads",
    "tma_info_system_dram_bw_use",
}

FORMULA_RE = re.compile(r"^metric expr (.*) for (\S+)$")
EVENT_RE = re.compile(r"^found event (.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", default="perf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-cpu", type=int, default=2)
    parser.add_argument("--events-per-pass", type=int, default=6)
    parser.add_argument("--system-events-per-pass", type=int, default=2)
    parser.add_argument("--mode", choices=("presets", "basic"), default="presets")
    return parser.parse_args()


def normalize_expression(value: str) -> str:
    return value.replace("\\", "")


def inspect_group(perf: str, group: str, system_cpu: int) -> str:
    command = [perf, "stat", "-vv"]
    if group in {"MemoryLat", "MemoryBW"}:
        command.extend(["-a", "-C", str(system_cpu)])
    command.extend(["-M", group, "--", "true"])
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=30
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out while expanding perf metric group {group}: {' '.join(command)}"
        ) from exc
    output = result.stdout + result.stderr
    if "metric expr" not in output or "found event" not in output:
        raise RuntimeError(
            f"Could not expand perf metric group {group}.\n"
            f"Command: {' '.join(command)}\n{output}"
        )
    return output


def is_system_event(event: str) -> bool:
    lowered = event.lower()
    return (
        event.startswith("UNC_")
        or lowered.startswith("arb@")
        or lowered.startswith("uncore_")
    )


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_basic_plan(output_dir: Path) -> None:
    write_csv(
        output_dir / "metric_targets.csv",
        ["source_group", "metric", "scope_class", "display_scale"],
        [
            {
                "source_group": "BasicEvents",
                "metric": metric,
                "scope_class": "core",
                "display_scale": 1.0,
            }
            for metric in BASIC_METRICS
        ],
    )
    write_csv(
        output_dir / "metric_formulas.csv",
        ["metric", "expression", "source_groups"],
        [
            {"metric": metric, "expression": expression, "source_groups": "BasicEvents"}
            for metric, expression in BASIC_METRICS.items()
        ],
    )
    write_csv(
        output_dir / "raw_event_catalog.csv",
        ["event", "scope_class"],
        [{"event": event, "scope_class": "core"} for event in BASIC_EVENTS],
    )
    pass_row = {
        "scope_class": "core",
        "pass_name": "basic_events",
        "event_count": len(BASIC_EVENTS),
        "events": "|".join(BASIC_EVENTS),
    }
    write_csv(
        output_dir / "pass_plan.csv",
        ["scope_class", "pass_name", "event_count", "events"],
        [pass_row],
    )
    (output_dir / "pass_plan.tsv").write_text(
        f"core\tbasic_events\t{len(BASIC_EVENTS)}\t{'|'.join(BASIC_EVENTS)}\n"
    )
    print("Basic target metrics: 3")
    print("Basic raw events: 6")
    print("Main replays per operator/regime/trial: 2 (user and kernel)")


def main() -> None:
    args = parse_args()
    if not 1 <= args.events_per_pass <= 6:
        raise ValueError("--events-per-pass must be between 1 and 6")
    if not 1 <= args.system_events_per_pass <= 6:
        raise ValueError("--system-events-per-pass must be between 1 and 6")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "basic":
        write_basic_plan(args.output_dir)
        return

    formulas: dict[str, str] = {}
    formula_groups: dict[str, set[str]] = {}
    events: set[str] = set()
    raw_outputs = args.output_dir / "perf_metric_expansion"
    raw_outputs.mkdir(exist_ok=True)

    for group in TARGET_METRICS:
        output = inspect_group(args.perf, group, args.system_cpu)
        (raw_outputs / f"{group}.txt").write_text(output)
        for line in output.splitlines():
            formula_match = FORMULA_RE.match(line)
            if formula_match:
                expression, name = formula_match.groups()
                expression = normalize_expression(expression)
                previous = formulas.setdefault(name, expression)
                if previous != expression:
                    raise RuntimeError(f"Conflicting formulas for {name}")
                formula_groups.setdefault(name, set()).add(group)
            event_match = EVENT_RE.match(line)
            if event_match:
                events.add(normalize_expression(event_match.group(1)))

    missing = sorted(
        metric
        for metrics in TARGET_METRICS.values()
        for metric in metrics
        if metric not in formulas
    )
    if missing:
        raise RuntimeError(f"Missing formulas for target metrics: {', '.join(missing)}")

    # duration_time is reconstructed from controlled JSON. perf's TSC token is
    # an expression-engine pseudo event and cannot be passed directly to -e;
    # the analyzer derives its task-active count from REF_TSC.
    events.discard("duration_time")
    events.discard("TSC")
    core_events = sorted(event for event in events if not is_system_event(event))
    system_events = sorted(event for event in events if is_system_event(event))

    target_rows = []
    for group, metrics in TARGET_METRICS.items():
        for metric in metrics:
            target_rows.append({
                "source_group": group,
                "metric": metric,
                "scope_class": "system" if metric in SYSTEM_METRICS else "core",
                "display_scale": 1.0 if metric.startswith("tma_info_") else 100.0,
            })
    write_csv(
        args.output_dir / "metric_targets.csv",
        ["source_group", "metric", "scope_class", "display_scale"],
        target_rows,
    )

    formula_rows = [
        {
            "metric": metric,
            "expression": expression,
            "source_groups": ";".join(sorted(formula_groups.get(metric, set()))),
        }
        for metric, expression in sorted(formulas.items())
    ]
    write_csv(
        args.output_dir / "metric_formulas.csv",
        ["metric", "expression", "source_groups"],
        formula_rows,
    )

    event_rows = [
        {"event": event, "scope_class": "system" if is_system_event(event) else "core"}
        for event in sorted(events)
    ]
    write_csv(
        args.output_dir / "raw_event_catalog.csv",
        ["event", "scope_class"],
        event_rows,
    )

    pass_rows = []
    for scope_class, selected, pass_size in (
        ("core", core_events, args.events_per_pass),
        ("system", system_events, args.system_events_per_pass),
    ):
        for index, pass_events in enumerate(chunks(selected, pass_size)):
            pass_rows.append({
                "scope_class": scope_class,
                "pass_name": f"raw_{scope_class}_{index:02d}",
                "event_count": len(pass_events),
                "events": "|".join(pass_events),
            })
    write_csv(
        args.output_dir / "pass_plan.csv",
        ["scope_class", "pass_name", "event_count", "events"],
        pass_rows,
    )
    with (args.output_dir / "pass_plan.tsv").open("w") as handle:
        for row in pass_rows:
            handle.write(
                f"{row['scope_class']}\t{row['pass_name']}\t{row['event_count']}\t{row['events']}\n"
            )

    print(f"Target metric slots: {len(target_rows)}")
    print(f"Unique target metrics: {len({row['metric'] for row in target_rows})}")
    print(f"Core raw events: {len(core_events)}")
    print(f"System raw events: {len(system_events)}")
    print(
        f"Raw passes: {len(pass_rows)} "
        f"(core <= {args.events_per_pass}, system <= {args.system_events_per_pass} events/pass)"
    )
    print(
        "Minimum main replays per operator/regime/trial: "
        f"{2 * math.ceil(len(core_events) / args.events_per_pass) + math.ceil(len(system_events) / args.system_events_per_pass)}"
    )


if __name__ == "__main__":
    main()
