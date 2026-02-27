import os
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from hta.analyzers.breakdown_analysis import KernelType, get_kernel_type
from hta.common.trace import Trace

TRACE_SUFFIXES = (".pt.trace.json", ".pt.trace.json.gz")
WORKER_RE = re.compile(r"worker_(?P<rank>\d+)\.(?P<ts>\d+)\.pt\.trace\.json")
RANK_DIR_RE = re.compile(r"rank_(\d+)")


def find_trace_files(trace_dir: Path) -> list[Path]:
    return sorted([p for p in trace_dir.rglob("*") if p.name.endswith(TRACE_SUFFIXES)])


def parse_rank(path: Path) -> int | None:
    for part in path.parts:
        match = RANK_DIR_RE.match(part)
        if match:
            return int(match.group(1))
    match = WORKER_RE.search(path.name)
    if match:
        return int(match.group("rank"))
    return None


def parse_timestamp(path: Path) -> int:
    match = WORKER_RE.search(path.name)
    if match:
        return int(match.group("ts"))
    return path.stat().st_mtime_ns


def select_latest_per_rank(files: Iterable[Path]) -> dict[int, str]:
    latest: dict[int, tuple[int, Path]] = {}
    for path in files:
        rank = parse_rank(path)
        if rank is None:
            continue
        ts = parse_timestamp(path)
        current = latest.get(rank)
        if current is None or ts > current[0]:
            latest[rank] = (ts, path)
    return {rank: str(path.resolve()) for rank, (_, path) in latest.items()}


def group_by_timestamp(files: Iterable[Path]) -> dict[int, dict[int, str]]:
    grouped: dict[int, dict[int, str]] = {}
    for path in files:
        rank = parse_rank(path)
        if rank is None:
            continue
        ts = parse_timestamp(path)
        grouped.setdefault(ts, {})[rank] = str(path.resolve())
    return grouped


def infer_out_dir(
    trace_dir: Path | None,
    trace_files: list[Path],
    reports_root: str,
    default_leaf: str,
) -> Path:
    if trace_files:
        common = Path(os.path.commonpath([str(p) for p in trace_files]))
    elif trace_dir is not None:
        common = trace_dir
    else:
        return Path(reports_root, default_leaf)

    if common.is_file():
        common = common.parent

    parts = list(common.parts)
    if "profile" in parts:
        idx = parts.index("profile")
        cut_idx = len(parts)
        for i in range(idx + 1, len(parts)):
            if RANK_DIR_RE.match(parts[i]):
                cut_idx = i
                break
        rel_parts = parts[idx + 1 : cut_idx]
        return Path(reports_root, *rel_parts) if rel_parts else Path(reports_root, default_leaf)

    return Path(reports_root, common.name)


def write_df(df: pd.DataFrame | None, path: Path) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_kernel_type_totals(trace: Trace) -> pd.DataFrame:
    sym_table = trace.symbol_table.get_sym_table()
    rows = []
    for rank, trace_df in trace.traces.items():
        gpu_df = trace_df[trace_df["stream"].ne(-1)]
        names = gpu_df["name"].apply(lambda idx: sym_table[int(idx)] if pd.notna(idx) else "")
        kernel_types = names.map(get_kernel_type)
        totals = gpu_df["dur"].groupby(kernel_types).sum()
        total_kernel_time = float(totals.sum()) if not totals.empty else 0.0
        rows.append(
            {
                "rank": int(rank),
                "total_kernel_time_us": total_kernel_time,
                "compute_time_us": float(totals.get(KernelType.COMPUTATION.name, 0.0)),
                "comm_time_us": float(totals.get(KernelType.COMMUNICATION.name, 0.0)),
                "memory_time_us": float(totals.get(KernelType.MEMORY.name, 0.0)),
            }
        )
    return pd.DataFrame(rows)
