import argparse
from pathlib import Path

import matplotlib


DEFAULT_TRAJ_PATH = R"D:\output\trajectory_opt.txt"
DEFAULT_LOG_PATH = R"D:\output\slam_post_debug_round2.log"
DEFAULT_OUTPUT_PATH = R"D:\output\constraint_edges_3d_from_script.png"
DEFAULT_CSV_PATH = R"D:\output\constraint_edges_3d_from_script.csv"


def detect_text_encoding(file_path: Path) -> str:
    raw_prefix = file_path.read_bytes()[:4]
    if raw_prefix.startswith(b"\xff\xfe"):
        return "utf-16"
    if raw_prefix.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw_prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a slam_post log and trajectory file, then visualize 3D pose-graph constraints with matplotlib."
    )
    parser.add_argument(
        "--traj",
        default=DEFAULT_TRAJ_PATH,
        help="Path to trajectory_opt.txt or another trajectory text file.",
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help="Path to slam_post log file containing 'BTC constraint:' lines.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Optional output image path. If provided, the figure is saved.",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help="Optional CSV output path for all parsed edges.",
    )
    parser.add_argument(
        "--adj-sample-step",
        type=int,
        default=1,
        help="Plot one adjacent edge every N edges to reduce clutter. Default: 1.",
    )
    parser.add_argument(
        "--hide-adjacent",
        action="store_true",
        help="Hide adjacent edges and show only loop-closure edges.",
    )
    parser.add_argument(
        "--hide-trajectory",
        action="store_true",
        help="Hide the trajectory polyline and point cloud.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display and only save outputs.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=28.0,
        help="3D view elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-58.0,
        help="3D view azimuth angle.",
    )
    return parser.parse_args()


ARGS = parse_args()

if not ARGS.show and not ARGS.no_show:
    ARGS.show = True

if ARGS.no_show:
    ARGS.show = False

if not ARGS.show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_positions(traj_path: Path) -> np.ndarray:
    positions: list[list[float]] = []

    for line in traj_path.read_text(encoding=detect_text_encoding(traj_path), errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        if len(parts) >= 11:
            first = float(parts[0])
            last = float(parts[10])
            if abs(last) > 1e8 and abs(first) < 1e8:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                continue
            if abs(first) > 1e8:
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

        if len(parts) >= 8:
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.asarray(positions, dtype=float)


def parse_loop_edges(log_path: Path) -> list[dict]:
    encoding = detect_text_encoding(log_path)
    loop_rows: list[dict] = []
    for line in log_path.read_text(encoding=encoding, errors="ignore").splitlines():
        if "BTC constraint: submap " not in line:
            continue

        payload = line.split("BTC constraint: submap ", 1)[1]
        left, right = payload.split(" <-> ", 1)
        matched, rest = right.split(" (score=", 1)
        score_str, rest = rest.split(", time=", 1)
        time_str, rest = rest.split("s, prior_dist=", 1)
        prior_dist_str, rest = rest.split("m, stds=", 1)
        stds_str = rest.rstrip(")")

        loop_rows.append(
            {
                "start_idx": int(left.strip()),
                "end_idx": int(matched.strip()),
                "score": float(score_str),
                "time_diff_s": float(time_str),
                "prior_dist_m": float(prior_dist_str),
                "stds": int(stds_str),
            }
        )
    return loop_rows


def build_edge_table(positions: np.ndarray, loop_rows: list[dict]) -> pd.DataFrame:
    edge_rows: list[dict] = []
    for idx in range(len(positions) - 1):
        start = positions[idx]
        end = positions[idx + 1]
        edge_rows.append(
            {
                "edge_type": "adjacent",
                "start_idx": idx,
                "end_idx": idx + 1,
                "start_x": start[0],
                "start_y": start[1],
                "start_z": start[2],
                "end_x": end[0],
                "end_y": end[1],
                "end_z": end[2],
                "length_m": float(np.linalg.norm(end - start)),
                "score": np.nan,
                "time_diff_s": np.nan,
                "prior_dist_m": np.nan,
                "stds": np.nan,
            }
        )

    for loop_row in loop_rows:
        start_idx = loop_row["start_idx"]
        end_idx = loop_row["end_idx"]
        if start_idx >= len(positions) or end_idx >= len(positions):
            continue

        start = positions[start_idx]
        end = positions[end_idx]
        edge_rows.append(
            {
                "edge_type": "btc_loop",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_x": start[0],
                "start_y": start[1],
                "start_z": start[2],
                "end_x": end[0],
                "end_y": end[1],
                "end_z": end[2],
                "length_m": float(np.linalg.norm(end - start)),
                "score": loop_row["score"],
                "time_diff_s": loop_row["time_diff_s"],
                "prior_dist_m": loop_row["prior_dist_m"],
                "stds": loop_row["stds"],
            }
        )

    return pd.DataFrame(edge_rows)


def set_equal_3d_axes(ax, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(maxs - mins) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_edges(positions: np.ndarray, edges: pd.DataFrame) -> plt.Figure:
    adjacent_edges = edges[edges["edge_type"] == "adjacent"].reset_index(drop=True)
    loop_edges = edges[edges["edge_type"] == "btc_loop"].reset_index(drop=True)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    if not ARGS.hide_trajectory:
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color="#808080",
            alpha=0.35,
            linewidth=1.0,
        )
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=np.arange(len(positions)),
            cmap="viridis",
            s=8,
            alpha=0.8,
        )

    if not ARGS.hide_adjacent:
        step = max(1, ARGS.adj_sample_step)
        for _, row in adjacent_edges.iloc[::step].iterrows():
            ax.plot(
                [row["start_x"], row["end_x"]],
                [row["start_y"], row["end_y"]],
                [row["start_z"], row["end_z"]],
                color="#1f77b4",
                alpha=0.35,
                linewidth=1.0,
            )

    for _, row in loop_edges.iterrows():
        ax.plot(
            [row["start_x"], row["end_x"]],
            [row["start_y"], row["end_y"]],
            [row["start_z"], row["end_z"]],
            color="#d62728",
            alpha=0.95,
            linewidth=2.0,
        )

    if not loop_edges.empty:
        ax.scatter(loop_edges["start_x"], loop_edges["start_y"], loop_edges["start_z"], color="#d62728", s=20, alpha=0.9)
        ax.scatter(loop_edges["end_x"], loop_edges["end_y"], loop_edges["end_z"], color="#ff9896", s=20, alpha=0.9)

    set_equal_3d_axes(ax, positions)
    ax.set_title("3D Pose Graph Constraints", pad=24, fontsize=18)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=ARGS.elev, azim=ARGS.azim)

    summary = (
        f"submaps={len(positions)}  adjacent={len(adjacent_edges)}  btc_loops={len(loop_edges)}\n"
        f"adj_mean={adjacent_edges['length_m'].mean():.2f}m  "
        f"loop_mean={(loop_edges['length_m'].mean() if not loop_edges.empty else 0.0):.2f}m  "
        f"loop_max={(loop_edges['length_m'].max() if not loop_edges.empty else 0.0):.2f}m"
    )
    fig.text(
        0.02,
        0.02,
        summary,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    return fig


def main() -> None:
    traj_path = Path(ARGS.traj)
    log_path = Path(ARGS.log)

    positions = load_positions(traj_path)
    loop_rows = parse_loop_edges(log_path)
    edges = build_edge_table(positions, loop_rows)

    if ARGS.csv:
        Path(ARGS.csv).parent.mkdir(parents=True, exist_ok=True)
        edges.to_csv(ARGS.csv, index=False)
        print(f"Wrote CSV: {ARGS.csv}")

    fig = plot_edges(positions, edges)

    if ARGS.output:
        Path(ARGS.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ARGS.output, dpi=200, bbox_inches="tight")
        print(f"Wrote image: {ARGS.output}")

    if ARGS.show:
        plt.show()


if __name__ == "__main__":
    main()