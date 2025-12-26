import argparse
import os
import re
import sys
import warnings
from collections.abc import Iterable
from math import ceil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from hta.analyzers.breakdown_analysis import BreakdownAnalysis, KernelType, get_kernel_type
from hta.analyzers.communication_analysis import CommunicationAnalysis
from hta.common.trace import Trace
from plotly.subplots import make_subplots

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


def infer_out_dir(trace_dir: Path | None, trace_files: list[Path]) -> Path:
    if trace_files:
        common = Path(os.path.commonpath([str(p) for p in trace_files]))
    elif trace_dir is not None:
        common = trace_dir
    else:
        return Path("reports", "profile")

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
        return Path("reports", *rel_parts) if rel_parts else Path("reports", "profile")

    return Path("reports", common.name)


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
        row = {
            "rank": int(rank),
            "total_kernel_time_us": total_kernel_time,
            "compute_time_us": float(totals.get(KernelType.COMPUTATION.name, 0.0)),
            "comm_time_us": float(totals.get(KernelType.COMMUNICATION.name, 0.0)),
            "memory_time_us": float(totals.get(KernelType.MEMORY.name, 0.0)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_kernel_name_totals(trace: Trace, top_k: int = 10) -> pd.DataFrame:
    sym_table = trace.symbol_table.get_sym_table()
    overall = None
    for trace_df in trace.traces.values():
        gpu_df = trace_df[trace_df["stream"].ne(-1)]
        grouped = gpu_df.groupby("name")["dur"].sum()
        grouped.index = grouped.index.map(lambda idx: sym_table[int(idx)] if pd.notna(idx) else "")
        grouped = grouped[grouped.index != ""]
        overall = grouped if overall is None else overall.add(grouped, fill_value=0.0)

    if overall is None or overall.empty:
        return pd.DataFrame(columns=["name", "total_time_us", "kernel_type"])

    df = overall.sort_values(ascending=False).head(top_k).reset_index()
    df.columns = ["name", "total_time_us"]
    df["kernel_type"] = df["name"].map(get_kernel_type)
    return df


def build_nccl_totals(trace: Trace) -> pd.DataFrame:
    sym_table = trace.symbol_table.get_sym_table()
    rows = []
    for rank, trace_df in trace.traces.items():
        gpu_df = trace_df[trace_df["stream"].ne(-1)]
        names = gpu_df["name"].apply(lambda idx: sym_table[int(idx)] if pd.notna(idx) else "")
        mask = names.str.contains("nccl", case=False, na=False) | names.str.contains(
            "allreduce", case=False, na=False
        )
        all_reduce_time = float(gpu_df.loc[mask, "dur"].sum())
        all_reduce_calls = int(mask.sum())
        rows.append(
            {
                "rank": int(rank),
                "all_reduce_time_us": all_reduce_time,
                "all_reduce_calls": all_reduce_calls,
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    comm_overlap: pd.DataFrame,
    temporal: pd.DataFrame,
    kernel_totals: pd.DataFrame,
    nccl_totals: pd.DataFrame,
) -> pd.DataFrame:
    temporal = temporal.rename(
        columns={
            "idle_time(us)": "idle_time_us",
            "compute_time(us)": "temporal_compute_time_us",
            "non_compute_time(us)": "non_compute_time_us",
            "kernel_time(us)": "kernel_time_us",
        }
    )
    comm_overlap = comm_overlap.rename(
        columns={
            "comp_comm_overlap_pctg": "comm_comp_overlap_pctg",
            "comp_comm_overlap_ratio": "comm_comp_overlap_pctg",
        }
    )
    summary = temporal.merge(comm_overlap, on="rank", how="left")
    summary = summary.merge(kernel_totals, on="rank", how="left")
    summary = summary.merge(nccl_totals, on="rank", how="left")
    summary["comm_overhead_pctg"] = summary.apply(
        lambda row: (row["comm_time_us"] / row["total_kernel_time_us"] * 100.0)
        if row.get("total_kernel_time_us", 0.0)
        else 0.0,
        axis=1,
    )
    summary["all_reduce_time_ms"] = summary["all_reduce_time_us"] / 1000.0
    summary["compute_time_ms"] = summary["temporal_compute_time_us"] / 1000.0
    summary["comm_comp_overlap_pctg"] = summary["comm_comp_overlap_pctg"].fillna(0.0)
    return summary


def write_dashboard(
    summary: pd.DataFrame, kernel_name_totals: pd.DataFrame, out_path: Path
) -> None:
    summary = summary.sort_values("rank")
    ranks = summary["rank"].astype(str).tolist()
    num_ranks = len(ranks)
    per_rank_cols = 3
    per_rank_rows = ceil(num_ranks / per_rank_cols) if num_ranks else 0
    kernel_type_colors = {
        "Computation": "#2563eb",
        "Communication": "#f97316",
        "Memory": "#10b981",
        "Other": "#9ca3af",
    }
    temporal_colors = {
        "Computation": "#2563eb",
        "Non-Compute": "#f97316",
        "Idle": "#9ca3af",
    }

    specs = [
        [{}, {}, {}],
        [{}, {}, {}],
        [{"colspan": 3}, None, None],
        [{"colspan": 3}, None, None],
    ]
    titles = [
        "All-Reduce Time (ms)",
        "Comm/Comp Overlap (%)",
        "Comm Overhead (%)",
        "Compute Time (ms)",
        "Compute Utilization (%)",
        "All-Reduce Calls",
        "Temporal Breakdown (%)",
        "Top Kernels (by GPU time)",
    ]

    for row_idx in range(per_rank_rows):
        row_specs = []
        for col_idx in range(per_rank_cols):
            rank_idx = row_idx * per_rank_cols + col_idx
            if rank_idx < num_ranks:
                row_specs.append({"type": "domain"})
                titles.append(f"Kernel Types (Rank {ranks[rank_idx]})")
            else:
                row_specs.append(None)
        specs.append(row_specs)

    fig = make_subplots(
        rows=len(specs),
        cols=3,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )

    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["all_reduce_time_ms"],
            name="All-Reduce Time",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["comm_comp_overlap_pctg"],
            name="Overlap",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["comm_overhead_pctg"],
            name="Comm Overhead",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["compute_time_ms"],
            name="Compute Time",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["compute_time_pctg"],
            name="Compute Utilization",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["all_reduce_calls"],
            name="All-Reduce Calls",
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["compute_time_pctg"],
            name="Computation",
            marker_color=temporal_colors["Computation"],
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["non_compute_time_pctg"],
            name="Non-Compute",
            marker_color=temporal_colors["Non-Compute"],
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=summary["idle_time_pctg"],
            name="Idle",
            marker_color=temporal_colors["Idle"],
        ),
        row=3,
        col=1,
    )
    if not kernel_name_totals.empty:
        top_kernels = kernel_name_totals.copy()
        top_kernels["total_time_ms"] = top_kernels["total_time_us"] / 1000.0
        top_kernels["label"] = top_kernels["name"].apply(
            lambda name: name if len(name) <= 60 else f"{name[:57]}..."
        )
        top_kernels = top_kernels.sort_values("total_time_ms", ascending=True)
        fig.add_trace(
            go.Bar(
                x=top_kernels["total_time_ms"],
                y=top_kernels["label"],
                orientation="h",
                marker_color=[
                    kernel_type_colors.get(kernel_type, kernel_type_colors["Other"])
                    for kernel_type in top_kernels["kernel_type"]
                ],
                customdata=top_kernels[["name", "kernel_type"]],
                hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{x:.2f} ms<extra></extra>",
                name="Top Kernels",
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    for rank_idx, rank in enumerate(ranks):
        row = 5 + (rank_idx // per_rank_cols)
        col = (rank_idx % per_rank_cols) + 1
        rank_row = summary[summary["rank"].astype(str) == rank].iloc[0]
        rank_kernel_totals = {
            "Computation": rank_row["compute_time_us"],
            "Communication": rank_row["comm_time_us"],
            "Memory": rank_row["memory_time_us"],
        }
        fig.add_trace(
            go.Pie(
                labels=list(rank_kernel_totals.keys()),
                values=list(rank_kernel_totals.values()),
                hole=0.35,
                marker=dict(colors=[kernel_type_colors[k] for k in rank_kernel_totals]),
                hovertemplate="%{label}<br>%{value:.2f} us (%{percent:.1%})<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="HTA Trace Summary",
        barmode="stack",
        showlegend=True,
        height=1300 + (per_rank_rows * 260),
        margin=dict(l=260, r=60, t=80, b=60),
        width=1300,
    )
    fig.update_xaxes(title_text="GPU Time (ms)", row=4, col=1)
    fig.update_yaxes(title_text="Kernel", row=4, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)


def print_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.sort_values("rank")
    overall = summary.mean(numeric_only=True)

    def pct(value: float) -> str:
        return f"{value:.2f}%"

    def ms(value: float) -> str:
        return f"{value:.2f} ms"

    print(f"Trace summary: {out_dir}")
    print("Overall (mean across ranks):")
    print(f"- All-reduce time: {ms(float(overall.get('all_reduce_time_ms', 0.0)))}")
    print(f"- Communication overhead: {pct(float(overall.get('comm_overhead_pctg', 0.0)))}")
    print(f"- Comm/comp overlap: {pct(float(overall.get('comm_comp_overlap_pctg', 0.0)))}")
    print(f"- Compute time: {ms(float(overall.get('compute_time_ms', 0.0)))}")
    print(f"- Compute utilization: {pct(float(overall.get('compute_time_pctg', 0.0)))}")
    print(f"- All-reduce calls: {float(overall.get('all_reduce_calls', 0.0)):.1f}")
    print("Per-rank:")
    for _, row in summary.iterrows():
        rank = int(row.get("rank", -1))
        print(
            "  "
            f"rank {rank}: all_reduce={ms(float(row.get('all_reduce_time_ms', 0.0)))} "
            f"comm_overhead={pct(float(row.get('comm_overhead_pctg', 0.0)))} "
            f"overlap={pct(float(row.get('comm_comp_overlap_pctg', 0.0)))} "
            f"compute={ms(float(row.get('compute_time_ms', 0.0)))} "
            f"util={pct(float(row.get('compute_time_pctg', 0.0)))} "
            f"calls={int(row.get('all_reduce_calls', 0))}"
        )


def analyze_run(
    trace_files: dict[int, str],
    out_dir: Path,
    include_last_step: bool,
    use_multiprocessing: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = Trace(trace_files=trace_files, trace_dir=".")
    trace.load_traces(
        include_last_profiler_step=include_last_step,
        use_multiprocessing=use_multiprocessing,
        use_memory_profiling=False,
    )

    comm_overlap = CommunicationAnalysis.get_comm_comp_overlap(trace, visualize=False)
    temporal = BreakdownAnalysis.get_temporal_breakdown(trace, visualize=False)
    kernel_totals = build_kernel_type_totals(trace)
    nccl_totals = build_nccl_totals(trace)
    summary = build_summary(comm_overlap, temporal, kernel_totals, nccl_totals)

    kernel_name_totals = build_kernel_name_totals(trace, top_k=12)
    write_df(summary, out_dir / "summary.csv")
    write_dashboard(summary, kernel_name_totals, out_dir / "summary.html")
    print_summary(summary, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze PyTorch profiler traces with HTA.")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("profile"),
        help="Directory to search for trace files.",
    )
    parser.add_argument(
        "--trace-files",
        nargs="*",
        type=Path,
        help="Explicit trace files to analyze (overrides --trace-dir).",
    )
    parser.add_argument(
        "--select",
        choices=["latest", "all"],
        default="latest",
        help="Select latest trace per rank or analyze all trace windows grouped by timestamp.",
    )
    parser.add_argument(
        "--include-last-step",
        action="store_true",
        help="Include the last profiler step when parsing traces.",
    )
    parser.add_argument(
        "--enable-multiprocessing",
        action="store_true",
        help="Use multiprocessing during trace parsing.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning, module="hta")

    if args.trace_files:
        trace_files = [p for p in args.trace_files if p.name.endswith(TRACE_SUFFIXES)]
    else:
        trace_files = find_trace_files(args.trace_dir)

    if not trace_files:
        print("No trace files found.", file=sys.stderr)
        return 1

    out_dir = infer_out_dir(args.trace_dir if not args.trace_files else None, trace_files)

    if args.select == "latest":
        trace_map = select_latest_per_rank(trace_files)
        if not trace_map:
            print("No trace files found with rank information.", file=sys.stderr)
            return 1
        analyze_run(
            trace_map,
            out_dir,
            args.include_last_step,
            args.enable_multiprocessing,
        )
        return 0

    grouped = group_by_timestamp(trace_files)
    if not grouped:
        print("No trace files found with timestamp information.", file=sys.stderr)
        return 1

    for idx, (ts, trace_map) in enumerate(sorted(grouped.items()), start=1):
        run_dir = out_dir / f"run_{idx}_{ts}"
        analyze_run(
            trace_map,
            run_dir,
            args.include_last_step,
            args.enable_multiprocessing,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
