"""
Lifetime K-Means — Split by review count
=========================================
Single-review users (n=1):
  Position on [0,1] lifetime timeline → density analysis + K-Means (1D binning).
  Anomaly signal: coordinated spike at a specific position (not uniform).

Multi-review users (n>1):
  50-bin probability vector of review density over lifetime → K-Means.
  Anomaly signal: burst near 0, periodic spikes, abnormal shape.

Global normalization: elapsed / max(elapsed across all users)
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import gaussian_kde

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
BASE_DIR   = Path(__file__).parents[1]
SINGLE_ANALYSIS = BASE_DIR / "analysis" / "single_review"
MULTI_ANALYSIS  = BASE_DIR / "analysis" / "multi_review"
SINGLE_RESULT   = BASE_DIR / "result" / "single_review"
MULTI_RESULT    = BASE_DIR / "result" / "multi_review"

N_BINS       = 50
N_CLUSTERS   = 20
SAMPLE_PER_CLUSTER = 3


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(con) -> tuple[pd.DataFrame, float]:
    logger.info("[Loading] Review elapsed times")
    df = con.execute("""
        SELECT r.user_id,
               EPOCH(r.date) - EPOCH(u.yelping_since) AS elapsed_sec
        FROM review r
        JOIN user u ON r.user_id = u.user_id
        WHERE r.date >= u.yelping_since
    """).fetchdf()
    global_max = float(df["elapsed_sec"].max())
    df["position"] = (df["elapsed_sec"] / global_max).clip(0, 1)
    logger.info(
        f"[Success] {len(df):,} reviews | {df['user_id'].nunique():,} users | "
        f"global_max={global_max/86400:.1f} days"
    )
    return df, global_max


def split_by_count(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df.groupby("user_id").size()
    single_ids = counts[counts == 1].index
    multi_ids  = counts[counts >  1].index
    logger.info(f"[Split] Single-review: {len(single_ids):,} | Multi-review: {len(multi_ids):,}")
    return df[df["user_id"].isin(single_ids)], df[df["user_id"].isin(multi_ids)]


# ════════════════════════════════════════════════════════════════════
# SINGLE-REVIEW ANALYSIS
# ════════════════════════════════════════════════════════════════════

def analyze_single(df_single: pd.DataFrame):
    logger.info("[Single] Starting analysis")
    positions = df_single["position"].values   # one value per user

    # ── KDE density plot ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Single-Review Users (n={len(positions):,})\n"
        "Distribution of review position on account lifetime [0=just joined → 1=dataset end]",
        fontsize=12
    )

    x_grid = np.linspace(0, 1, 500)
    kde    = gaussian_kde(positions, bw_method=0.03)
    density = kde(x_grid)

    ax = axes[0]
    ax.fill_between(x_grid, density, alpha=0.4, color="#4C72B0")
    ax.plot(x_grid, density, color="#4C72B0", lw=1.5)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="Uniform baseline")
    ax.set_xlabel("Lifetime position", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("KDE Density (bandwidth=0.03)", fontsize=10)
    ax.legend(fontsize=9)

    # ── Histogram (50 bins) ───────────────────────────────────────
    ax2 = axes[1]
    counts_hist, edges = np.histogram(positions, bins=N_BINS, range=(0, 1))
    expected = len(positions) / N_BINS
    colors = ["#DD8452" if c > expected * 2 else "#4C72B0" for c in counts_hist]
    ax2.bar(edges[:-1], counts_hist, width=1/N_BINS, align="edge", color=colors, alpha=0.85)
    ax2.axhline(expected, color="gray", ls="--", lw=1, label=f"Expected uniform ({expected:,.0f})")
    ax2.set_xlabel("Lifetime position", fontsize=10)
    ax2.set_ylabel("User count", fontsize=10)
    ax2.set_title("Histogram (orange = >2× expected)", fontsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(SINGLE_ANALYSIS / "position_distribution.png", dpi=150)
    plt.close()
    logger.info(f"[Saved] position_distribution.png")

    # ── K-Means on 50-bin one-hot vectors ────────────────────────
    df_single = df_single.copy()
    df_single["bin"] = (df_single["position"] * N_BINS).clip(0, N_BINS - 1).astype(int)
    vectors = np.zeros((len(df_single), N_BINS), dtype=np.float32)
    for i, b in enumerate(df_single["bin"].values):
        vectors[i, b] = 1.0

    logger.info(f"[Single KMeans] K={N_CLUSTERS} on {len(vectors):,} users")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42,
                         batch_size=65536, n_init=10)
    labels = km.fit_predict(vectors)

    result_df = pd.DataFrame({
        "user_id": df_single["user_id"].values,
        "position": df_single["position"].values,
        "cluster": labels,
    })
    result_df.to_parquet(SINGLE_RESULT / "single_assignments.parquet", index=False)

    # Cluster stats
    stats = result_df.groupby("cluster").agg(
        count=("user_id", "count"),
        mean_position=("position", "mean"),
        std_position=("position", "std"),
    ).reset_index().sort_values("mean_position")
    stats.to_csv(SINGLE_RESULT / "cluster_stats.csv", index=False)
    logger.info(f"\n[Single Cluster Stats]\n{stats.to_string(index=False)}")

    # Overview plot: centroid bars sorted by peak position
    centroids = km.cluster_centers_
    order  = np.argsort(centroids.argmax(axis=1))
    n_cols = 5
    n_rows = (N_CLUSTERS + n_cols - 1) // n_cols
    x      = np.linspace(0, 1, N_BINS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
    fig.suptitle(f"Single-Review Users — Cluster Centroids (K={N_CLUSTERS})", fontsize=12)

    for idx, cid in enumerate(order):
        ax  = axes[idx // n_cols][idx % n_cols]
        c   = centroids[cid]
        n   = (labels == cid).sum()
        pos = x[c.argmax()]
        ax.bar(x, c, width=1/N_BINS, color="#4C72B0", alpha=0.85, align="edge")
        ax.axvline(pos, color="red", lw=1, ls="--")
        ax.set_title(f"C{cid}  n={n:,}\npos={pos:.2f}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=6)

    for idx in range(N_CLUSTERS, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(SINGLE_ANALYSIS / "cluster_overview.png", dpi=150)
    plt.close()
    logger.info("[Saved] single cluster_overview.png")
    logger.info("[Single] Done")


# ════════════════════════════════════════════════════════════════════
# MULTI-REVIEW ANALYSIS
# ════════════════════════════════════════════════════════════════════

def compute_multi_vectors(df_multi: pd.DataFrame) -> tuple[np.ndarray, list]:
    df_multi = df_multi.copy()
    df_multi["bin"] = (df_multi["position"] * N_BINS).clip(0, N_BINS - 1).astype(int)

    counts = (
        df_multi.groupby(["user_id", "bin"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(N_BINS), fill_value=0)
    )
    row_totals = counts.sum(axis=1)
    vectors = counts.div(row_totals, axis=0).values.astype(np.float32)
    return vectors, counts.index.tolist()


def analyze_multi(df_multi: pd.DataFrame):
    logger.info("[Multi] Starting analysis")
    vectors, user_ids = compute_multi_vectors(df_multi)
    logger.info(f"[Multi] Vectors: {vectors.shape}")

    logger.info(f"[Multi KMeans] K={N_CLUSTERS} on {len(vectors):,} users")
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42,
                         batch_size=65536, n_init=10)
    labels = km.fit_predict(vectors)
    centroids = km.cluster_centers_

    result_df = pd.DataFrame({"user_id": user_ids, "cluster": labels})
    result_df.to_parquet(MULTI_RESULT / "multi_assignments.parquet", index=False)

    stats = result_df.groupby("cluster").size().reset_index(name="count")
    stats["entropy"]      = [-np.sum(centroids[c] * np.log2(centroids[c] + 1e-10))
                              for c in stats["cluster"]]
    stats["peak_position"] = centroids[stats["cluster"].values].argmax(axis=1) / N_BINS
    stats = stats.sort_values("entropy")
    stats.to_csv(MULTI_RESULT / "cluster_stats.csv", index=False)
    logger.info(f"\n[Multi Cluster Stats]\n{stats.to_string(index=False)}")

    # Overview
    x     = np.linspace(0, 1, N_BINS)
    order  = np.argsort(centroids.argmax(axis=1))
    n_cols = 5
    n_rows = (N_CLUSTERS + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
    fig.suptitle(
        f"Multi-Review Users — Cluster Centroids (K={N_CLUSTERS})\n"
        "X = account lifetime [0→1]  |  Y = review density",
        fontsize=12
    )
    for idx, cid in enumerate(order):
        ax  = axes[idx // n_cols][idx % n_cols]
        c   = centroids[cid]
        ent = -np.sum(c * np.log2(c + 1e-10))
        n   = (labels == cid).sum()
        pos = x[c.argmax()]
        ax.bar(x, c, width=1/N_BINS, color="#4C72B0", alpha=0.85, align="edge")
        ax.axvline(pos, color="red", lw=1, ls="--", alpha=0.7)
        ax.set_title(f"C{cid}  n={n:,}\nEntropy={ent:.2f}  peak={pos:.2f}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=6)

    for idx in range(N_CLUSTERS, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(MULTI_ANALYSIS / "cluster_overview.png", dpi=150)
    plt.close()
    logger.info("[Saved] multi cluster_overview.png")

    # Sample plots per cluster
    uid_to_pos = df_multi.groupby("user_id")["position"].apply(list).to_dict()
    logger.info("[Multi] Generating sample plots")
    for cid in range(N_CLUSTERS):
        cluster_users = result_df[result_df["cluster"] == cid]["user_id"].tolist()
        sample = cluster_users[:SAMPLE_PER_CLUSTER]
        _plot_multi_samples(cid, centroids[cid], sample, uid_to_pos)

    logger.info("[Multi] Done")


def _plot_multi_samples(cid, centroid, sample_ids, uid_to_pos):
    n   = len(sample_ids)
    x   = np.linspace(0, 1, N_BINS)
    fig = plt.figure(figsize=(4 * (n + 1), 4))
    gs  = gridspec.GridSpec(1, n + 1, figure=fig, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x, centroid, width=1/N_BINS, color="#4C72B0", alpha=0.85, align="edge")
    ax0.set_title(f"Cluster {cid}\ncentroid", fontsize=9)
    ax0.set_xlim(0, 1)
    ax0.set_xlabel("Lifetime pos", fontsize=8)

    for i, uid in enumerate(sample_ids):
        ax = fig.add_subplot(gs[0, i + 1])
        positions = uid_to_pos.get(uid, [])
        ax.hist(positions, bins=N_BINS, range=(0, 1), color="#DD8452", alpha=0.85)
        ax.set_title(f"uid={uid}\nn={len(positions)}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Lifetime pos", fontsize=8)

    plt.suptitle(f"Multi-Review Cluster {cid} — Centroid + Samples", fontsize=10)
    plt.tight_layout()
    plt.savefig(MULTI_ANALYSIS / f"cluster_{cid:02d}_samples.png", dpi=130)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    for d in [SINGLE_ANALYSIS, MULTI_ANALYSIS, SINGLE_RESULT, MULTI_RESULT]:
        d.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df, global_max = load_data(con)
    con.close()

    df_single, df_multi = split_by_count(df)

    analyze_single(df_single)
    analyze_multi(df_multi)

    logger.info("[All done]")


if __name__ == "__main__":
    main()
