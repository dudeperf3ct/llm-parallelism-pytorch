import argparse
import gzip
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from analyze_traces_common import (
    TRACE_SUFFIXES,
    build_kernel_type_totals,
    find_trace_files,
    group_by_timestamp,
    infer_out_dir,
    select_latest_per_rank,
    write_df,
)
from hta.analyzers.breakdown_analysis import BreakdownAnalysis
from hta.analyzers.communication_analysis import CommunicationAnalysis
from hta.common.trace import Trace
from plotly.subplots import make_subplots

CPU_SCOPE_MAP = {
    "pp.forward": "forward_cpu_us",
    "pp.backward": "backward_cpu_us",
    "pp.forward_warmup": "forward_warmup_cpu_us",
    "pp.1f1b_steady": "steady_1f1b_cpu_us",
    "pp.backward_drain": "backward_drain_cpu_us",
    "pp.comm.send": "send_cpu_us",
    "pp.comm.recv": "recv_cpu_us",
    "pp.optimizer_step": "optimizer_step_cpu_us",
    "pp.optimizer_zero_grad": "optimizer_zero_grad_cpu_us",
}


def _load_trace_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with path.open("rt") as f:
        return json.load(f)


def build_memory_summary(trace_files: dict[int, str]) -> pd.DataFrame:
    rows = []
    for rank, trace_path in trace_files.items():
        trace_json = _load_trace_json(Path(trace_path))
        events = trace_json.get("traceEvents", [])
        mem_rows = []
        for event in events:
            if event.get("name") != "[memory]":
                continue
            args = event.get("args", {})
            if not isinstance(args, dict) or args.get("Device Type") != 1:
                continue
            total_alloc = args.get("Total Allocated")
            total_res = args.get("Total Reserved")
            ts = event.get("ts")
            if ts is None or total_alloc is None or total_res is None:
                continue
            mem_rows.append(
                {
                    "ts": int(ts),
                    "total_allocated_bytes": int(total_alloc),
                    "total_reserved_bytes": int(total_res),
                }
            )

        if not mem_rows:
            rows.append(
                {
                    "rank": int(rank),
                    "memory_samples": 0,
                    "peak_allocated_mb": 0.0,
                    "peak_reserved_mb": 0.0,
                    "p50_allocated_mb": 0.0,
                    "p95_allocated_mb": 0.0,
                    "final_allocated_mb": 0.0,
                    "final_reserved_mb": 0.0,
                }
            )
            continue

        df = pd.DataFrame(mem_rows).sort_values("ts")
        alloc_mb = df["total_allocated_bytes"] / 1024**2
        reserved_mb = df["total_reserved_bytes"] / 1024**2
        rows.append(
            {
                "rank": int(rank),
                "memory_samples": int(len(df)),
                "peak_allocated_mb": float(alloc_mb.max()),
                "peak_reserved_mb": float(reserved_mb.max()),
                "p50_allocated_mb": float(alloc_mb.quantile(0.50)),
                "p95_allocated_mb": float(alloc_mb.quantile(0.95)),
                "final_allocated_mb": float(alloc_mb.iloc[-1]),
                "final_reserved_mb": float(reserved_mb.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def build_p2p_kernel_totals(trace: Trace) -> pd.DataFrame:
    sym_table = trace.symbol_table.get_sym_table()
    rows = []
    for rank, trace_df in trace.traces.items():
        gpu_df = trace_df[trace_df["stream"].ne(-1)].copy()
        names = gpu_df["name"].apply(lambda idx: sym_table[int(idx)] if pd.notna(idx) else "")
        mask = names.str.contains("sendrecv", case=False, na=False) | names.str.contains(
            "nccl:coalesced", case=False, na=False
        )
        rows.append(
            {
                "rank": int(rank),
                "p2p_kernel_time_us": float(gpu_df.loc[mask, "dur"].sum()),
                "p2p_kernel_calls": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_cpu_scope_totals(trace_files: dict[int, str]) -> pd.DataFrame:
    rows = []
    for rank, trace_path in trace_files.items():
        trace_json = _load_trace_json(Path(trace_path))
        row = {"rank": int(rank)}
        total_pp_cpu_us = 0.0
        for col in CPU_SCOPE_MAP.values():
            row[col] = 0.0

        for event in trace_json.get("traceEvents", []):
            name = event.get("name")
            dur = event.get("dur")
            if not isinstance(name, str) or dur is None:
                continue
            col = CPU_SCOPE_MAP.get(name)
            if col is not None:
                row[col] += float(dur)
            if name.startswith("pp."):
                total_pp_cpu_us += float(dur)

        row["total_pp_cpu_us"] = total_pp_cpu_us
        row["comm_cpu_us"] = row["send_cpu_us"] + row["recv_cpu_us"]
        row["forward_like_cpu_us"] = row["forward_cpu_us"] + row["forward_warmup_cpu_us"]
        row["backward_like_cpu_us"] = row["backward_cpu_us"] + row["backward_drain_cpu_us"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(
    comm_overlap: pd.DataFrame,
    temporal: pd.DataFrame,
    kernel_totals: pd.DataFrame,
    p2p_kernel_totals: pd.DataFrame,
    cpu_scope_totals: pd.DataFrame,
    memory_summary: pd.DataFrame,
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
    summary = summary.merge(p2p_kernel_totals, on="rank", how="left")
    summary = summary.merge(cpu_scope_totals, on="rank", how="left")
    summary = summary.merge(memory_summary, on="rank", how="left")
    summary["comm_overhead_pctg"] = summary.apply(
        lambda row: (
            (row["comm_time_us"] / row["total_kernel_time_us"] * 100.0)
            if row.get("total_kernel_time_us", 0.0)
            else 0.0
        ),
        axis=1,
    )
    summary["compute_time_ms"] = summary["temporal_compute_time_us"] / 1000.0
    summary["p2p_kernel_time_ms"] = summary["p2p_kernel_time_us"] / 1000.0
    summary["total_pp_cpu_ms"] = summary["total_pp_cpu_us"] / 1000.0
    summary["comm_cpu_ms"] = summary["comm_cpu_us"] / 1000.0
    summary["forward_like_cpu_ms"] = summary["forward_like_cpu_us"] / 1000.0
    summary["backward_like_cpu_ms"] = summary["backward_like_cpu_us"] / 1000.0
    summary["optimizer_cpu_ms"] = (
        summary["optimizer_step_cpu_us"] + summary["optimizer_zero_grad_cpu_us"]
    ) / 1000.0
    summary["comm_comp_overlap_pctg"] = summary["comm_comp_overlap_pctg"].fillna(0.0)
    summary["p2p_kernel_calls"] = summary["p2p_kernel_calls"].fillna(0).astype(int)
    return summary


def write_dashboard(summary: pd.DataFrame, out_path: Path) -> None:
    summary = summary.sort_values("rank")
    ranks = summary["rank"].astype(str).tolist()
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Peak CUDA Memory (MB)",
            "P2P GPU Kernel Time (ms)",
            "Temporal Breakdown (%)",
            "PP CPU Scope Time (ms)",
            "Communication Overhead (%)",
            "Compute vs Comm Time (ms)",
        ],
    )

    fig.add_trace(
        go.Bar(x=ranks, y=summary["peak_allocated_mb"], name="Peak Allocated"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["peak_reserved_mb"], name="Peak Reserved"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["p2p_kernel_time_ms"], name="P2P Kernel Time"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["compute_time_pctg"], name="Compute %"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["non_compute_time_pctg"], name="Non-Compute %"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["idle_time_pctg"], name="Idle %"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["forward_like_cpu_ms"], name="Forward CPU"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["backward_like_cpu_ms"], name="Backward CPU"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["comm_cpu_ms"], name="Send/Recv CPU"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["optimizer_cpu_ms"], name="Optimizer CPU"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["comm_overhead_pctg"], name="Comm Overhead %"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["compute_time_ms"], name="Compute Time"),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=ranks, y=summary["p2p_kernel_time_ms"], name="P2P Comm Time"),
        row=3,
        col=2,
    )

    fig.update_layout(
        title="HTA Pipeline Parallel Summary",
        barmode="stack",
        showlegend=True,
        height=1200,
        width=1500,
        margin=dict(l=80, r=30, t=80, b=60),
    )
    fig.update_yaxes(title_text="MB", row=1, col=1)
    fig.update_yaxes(title_text="ms", row=1, col=2)
    fig.update_yaxes(title_text="%", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="ms", row=2, col=2)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="ms", row=3, col=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # fig.write_html(out_path)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=False)


def print_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.sort_values("rank")
    overall = summary.mean(numeric_only=True)
    print(f"Pipeline trace summary: {out_dir}")
    print("Overall (mean across ranks):")
    print(f"- Peak allocated memory: {float(overall.get('peak_allocated_mb', 0.0)):.2f} MB")
    print(f"- Peak reserved memory: {float(overall.get('peak_reserved_mb', 0.0)):.2f} MB")
    print(f"- P2P kernel time: {float(overall.get('p2p_kernel_time_ms', 0.0)):.2f} ms")
    print(f"- P2P kernel calls: {float(overall.get('p2p_kernel_calls', 0.0)):.1f}")
    print(f"- Comm/comp overlap: {float(overall.get('comm_comp_overlap_pctg', 0.0)):.2f}%")
    print(f"- Communication overhead: {float(overall.get('comm_overhead_pctg', 0.0)):.2f}%")
    print("Per-rank:")
    for _, row in summary.iterrows():
        print(
            "  "
            f"rank {int(row['rank'])}: peak_alloc={float(row.get('peak_allocated_mb', 0.0)):.2f}MB "
            f"peak_res={float(row.get('peak_reserved_mb', 0.0)):.2f}MB "
            f"p2p={float(row.get('p2p_kernel_time_ms', 0.0)):.2f}ms "
            f"calls={int(row.get('p2p_kernel_calls', 0))} "
            f"cpu_pp={float(row.get('total_pp_cpu_ms', 0.0)):.2f}ms"
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
        use_memory_profiling=True,
    )

    comm_overlap = CommunicationAnalysis.get_comm_comp_overlap(trace, visualize=False)
    temporal = BreakdownAnalysis.get_temporal_breakdown(trace, visualize=False)
    kernel_totals = build_kernel_type_totals(trace)
    p2p_kernel_totals = build_p2p_kernel_totals(trace)
    cpu_scope_totals = build_cpu_scope_totals(trace_files)
    memory_summary = build_memory_summary(trace_files)
    summary = build_summary(
        comm_overlap,
        temporal,
        kernel_totals,
        p2p_kernel_totals,
        cpu_scope_totals,
        memory_summary,
    )

    write_df(summary, out_dir / "summary.csv")
    write_dashboard(summary, out_dir / "summary.html")
    print_summary(summary, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze pipeline parallel traces with HTA (P2P + memory focus)."
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("profile_pp"),
        help="Directory to search for PP trace files.",
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

    out_dir = infer_out_dir(
        args.trace_dir if not args.trace_files else None,
        trace_files,
        reports_root="reports_pp",
        default_leaf="profile_pp",
    )

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
        analyze_run(
            trace_map,
            out_dir / f"run_{idx}_{ts}",
            args.include_last_step,
            args.enable_multiprocessing,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
