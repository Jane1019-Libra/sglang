#!/usr/bin/env python3
"""Summarize SGLang phase trace markers by rank and iteration.

The markers are emitted as torch profiler record_function events whose names
look like:

    phase::cuda_graph.replay rank=0 pass_id=123 mode=DECODE

This script prints one CSV row per logical phase group and includes each rank's
start/end/duration plus cross-rank skew.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PHASE_PREFIX = "phase::"


def open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_events(path: Path) -> list[dict[str, Any]]:
    with open_trace(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError(f"unsupported trace JSON root: {type(data).__name__}")

    if not isinstance(events, list):
        raise ValueError("traceEvents is not a list")
    return events


def parse_phase_name(name: str) -> tuple[str, dict[str, str]]:
    payload = name[len(PHASE_PREFIX) :]
    parts = payload.split()
    phase = parts[0] if parts else ""
    fields: dict[str, str] = {}

    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value

    return phase, fields


def event_rank(event: dict[str, Any], fields: dict[str, str]) -> str | None:
    rank = fields.get("rank")
    if rank is not None:
        return rank

    args = event.get("args")
    if isinstance(args, dict):
        for key in ("rank", "tp_rank"):
            value = args.get(key)
            if value is not None:
                return str(value)
    return None


def group_key(phase: str, fields: dict[str, str]) -> tuple[str, str, str, str]:
    logical_id = fields.get("iter") or fields.get("pass_id") or fields.get("step") or ""
    logical_kind = (
        "iter"
        if "iter" in fields
        else "pass_id" if "pass_id" in fields else "step" if "step" in fields else ""
    )
    mode = fields.get("mode", "")
    extra = fields.get("key", "")
    return phase, logical_kind, logical_id, mode if not extra else f"{mode}|key={extra}"


def collect_phase_events(events: list[dict[str, Any]]):
    groups: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for event in events:
        name = event.get("name")
        if not isinstance(name, str) or not name.startswith(PHASE_PREFIX):
            continue
        if "ts" not in event:
            continue

        phase, fields = parse_phase_name(name)
        rank = event_rank(event, fields)
        if rank is None:
            continue

        ts = float(event["ts"])
        dur = float(event.get("dur", 0.0) or 0.0)
        groups[group_key(phase, fields)][rank].append(
            {
                "ts": ts,
                "dur": dur,
                "end": ts + dur,
                "pid": event.get("pid", ""),
                "tid": event.get("tid", ""),
                "name": name,
            }
        )

    return groups


def summarize(groups):
    rows: list[dict[str, Any]] = []

    for (phase, logical_kind, logical_id, mode), by_rank in groups.items():
        first_by_rank = {
            rank: min(events, key=lambda event: event["ts"])
            for rank, events in by_rank.items()
        }
        starts = [event["ts"] for event in first_by_rank.values()]
        ends = [event["end"] for event in first_by_rank.values()]
        durations = [event["dur"] for event in first_by_rank.values()]

        row: dict[str, Any] = {
            "phase": phase,
            "logical_kind": logical_kind,
            "logical_id": logical_id,
            "mode": mode,
            "rank_count": len(first_by_rank),
            "start_min_us": min(starts),
            "start_max_us": max(starts),
            "start_skew_us": max(starts) - min(starts),
            "end_min_us": min(ends),
            "end_max_us": max(ends),
            "end_skew_us": max(ends) - min(ends),
            "dur_min_us": min(durations),
            "dur_max_us": max(durations),
            "dur_spread_us": max(durations) - min(durations),
        }

        for rank in sorted(
            first_by_rank, key=lambda value: int(value) if value.isdigit() else value
        ):
            event = first_by_rank[rank]
            row[f"rank{rank}_start_us"] = event["ts"]
            row[f"rank{rank}_end_us"] = event["end"]
            row[f"rank{rank}_dur_us"] = event["dur"]

        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            row["logical_kind"],
            (
                int(row["logical_id"])
                if str(row["logical_id"]).isdigit()
                else row["logical_id"]
            ),
            row["phase"],
            row["mode"],
        ),
    )


def write_csv(rows: list[dict[str, Any]], output):
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "phase",
        "logical_kind",
        "logical_id",
        "mode",
        "rank_count",
        "start_skew_us",
        "end_skew_us",
        "dur_spread_us",
        "start_min_us",
        "start_max_us",
        "end_min_us",
        "end_max_us",
        "dur_min_us",
        "dur_max_us",
    ]
    for key in preferred:
        fieldnames.append(key)
        seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="Chrome trace JSON or JSON.GZ")
    parser.add_argument(
        "--min-ranks",
        type=int,
        default=2,
        help="only emit groups seen on at least this many ranks",
    )
    parser.add_argument(
        "--min-start-skew-us",
        type=float,
        default=0.0,
        help="only emit groups with at least this much start skew",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write CSV here instead of stdout",
    )
    args = parser.parse_args()

    events = load_events(args.trace)
    groups = collect_phase_events(events)
    rows = [
        row
        for row in summarize(groups)
        if row["rank_count"] >= args.min_ranks
        and row["start_skew_us"] >= args.min_start_skew_us
    ]

    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as f:
            write_csv(rows, f)
    else:
        write_csv(rows, sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
