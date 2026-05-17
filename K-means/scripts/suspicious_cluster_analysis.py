import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from loguru import logger

HOUR_COLS = [f"h{h:02d}" for h in range(24)]
N_YEARS = 18
N_CLUSTERS = 20
MORNING_HOURS = list(range(6, 12))   # 06-11: biological dip window
SUSPICIOUS = [10, 19]
HUMAN_REF  = [14, 16]               # lowest entropy clusters

matrix_path = Path(__file__).parents[2] / "user behavier matrix" / "result" / "user_hour_matrix.parquet"
assign_path = Path(__file__).parents[1] / "analysis" / "kmeans_clustering" / "kmeans_assignments.parquet"
stats_path  = Path(__file__).parents[1] / "analysis" / "kmeans_clustering" / "cluster_stats.csv"
output_dir  = Path(__file__).parents[1] / "analysis" / "kmeans_clustering"


# ── 1. Morning activity ratio per cluster ─────────────────────────────────────

def compute_morning_ratio(centroids_df: pd.DataFrame) -> pd.DataFrame:
    """centroids_df: index=cluster, columns=h00..h23 (centroids from stats csv lacks this)
    We recompute from assignments + matrix."""
    pass


# ── helpers ───────────────────────────────────────────────────────────────────

def get_mean_matrix(hour_data: pd.DataFrame, user_ids: list) -> np.ndarray:
    """Return (N_YEARS, 24) mean matrix for given user_ids."""
    mask = hour_data.index.get_level_values("user_id").isin(set(user_ids))
    sub = hour_data[mask].copy()
    sub[sub == -1] = np.nan
    grp = sub.groupby(level="year")[HOUR_COLS].mean().sort_index()
    # year index is actual calendar years (e.g. 2005-2022); pad to N_YEARS rows
    all_years = grp.index.tolist()
    if all_years:
        year_min, year_max = min(all_years), max(all_years)
        full_range = range(year_min, year_min + N_YEARS)
        grp = grp.reindex(full_range)
    return grp.values  # (18, 24)


def get_behavior_vector(hour_data: pd.DataFrame, user_ids: list) -> np.ndarray:
    """Return mean 24-dim normalized behavior vector for user set."""
    mask = hour_data.index.get_level_values("user_id").isin(set(user_ids))
    sub = hour_data[mask].copy()
    sub[sub == -1] = np.nan
    # Same algorithm: per-year normalize → sum → re-normalize
    eps = 1e-10
    row_totals = sub[HOUR_COLS].sum(axis=1)
    active = row_totals > 0
    normed = sub[HOUR_COLS].copy().astype(float)
    normed[active] = normed[active].div(row_totals[active], axis=0)
    normed[~active] = 0.0
    normed = normed.fillna(0.0)
    accumulated = normed.groupby(level="user_id").sum()
    user_totals = accumulated.sum(axis=1)
    bv = accumulated.div(user_totals + eps, axis=0)
    return bv.mean(axis=0).values  # (24,)


def sample_user_matrices(hour_data: pd.DataFrame, user_ids: list, n=6, seed=42) -> list:
    """Return list of (uid, 18×24 array) for n sampled users."""
    rng = np.random.default_rng(seed)
    sampled = rng.choice(user_ids, size=min(n, len(user_ids)), replace=False).tolist()
    results = []
    for uid in sampled:
        try:
            mat = hour_data.loc[uid][HOUR_COLS].values.astype(float)
            mat = np.where(mat == -1, np.nan, mat)
            results.append((uid, mat))
        except Exception:
            pass
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("[Loading] Data")
    assignments = pd.read_parquet(assign_path)
    stats = pd.read_csv(stats_path)

    matrix = pd.read_parquet(matrix_path)
    logger.info(f"[Success] Matrix: {matrix.shape}")

    hour_data = matrix[HOUR_COLS].copy()
    uid_to_cluster = assignments.set_index("user_id")["cluster"]

    # ── Plot 1: Morning activity ratio bar chart ──────────────────────────────
    logger.info("[Computing] Morning ratio per cluster")

    morning_cols = [f"h{h:02d}" for h in MORNING_HOURS]
    valid_ids = set(assignments["user_id"])
    mask = hour_data.index.get_level_values("user_id").isin(valid_ids)
    hd = hour_data[mask].copy().astype(float)
    hd[hd == -1] = np.nan

    # User-level: sum across years, compute morning fraction
    user_sum = hd.groupby(level="user_id")[HOUR_COLS].sum()
    total = user_sum.sum(axis=1)
    morning_frac = user_sum[morning_cols].sum(axis=1) / (total + 1e-10)
    morning_frac = morning_frac.rename("morning_ratio")

    cluster_morning = (
        morning_frac.reset_index()
        .rename(columns={"user_id": "user_id"})
        .merge(assignments, on="user_id")
        .groupby("cluster")["morning_ratio"]
        .mean()
        .reset_index()
        .merge(stats[["cluster", "entropy", "peak_hour", "count"]], on="cluster")
        .sort_values("morning_ratio", ascending=False)
    )
    logger.info(f"\n[Morning Ratio by Cluster]\n{cluster_morning.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#DD8452" if c in SUSPICIOUS else "#4C72B0" for c in cluster_morning["cluster"]]
    bars = ax.bar(cluster_morning["cluster"].astype(str), cluster_morning["morning_ratio"] * 100, color=colors)
    ax.axhline(1/24 * 100 * len(MORNING_HOURS), color="gray", ls="--", lw=1,
               label="Expected if uniform (25%)")
    ax.set_xlabel("Cluster (sorted by morning activity ratio)", fontsize=10)
    ax.set_ylabel("Mean 06-11h Activity (%)", fontsize=10)
    ax.set_title("Cluster Morning Activity Ratio (06:00–11:59)\nOrange = Suspicious | Dashed = Uniform baseline", fontsize=11)
    ax.legend()
    # Annotate suspicious clusters
    for bar, (_, row) in zip(bars, cluster_morning.iterrows()):
        if row["cluster"] in SUSPICIOUS:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"C{int(row['cluster'])}", ha="center", va="bottom", fontsize=8, color="#DD8452", fontweight="bold")
    plt.tight_layout()
    p = output_dir / "morning_ratio_by_cluster.png"
    plt.savefig(p, dpi=150)
    plt.close()
    logger.info(f"[Saved] {p}")

    # ── Plot 2: Side-by-side behavior vectors + mean matrices ─────────────────
    target_clusters = SUSPICIOUS + HUMAN_REF
    cluster_users = {c: assignments[assignments["cluster"] == c]["user_id"].tolist()
                     for c in target_clusters}

    logger.info("[Computing] Behavior vectors and mean matrices")
    bvectors = {c: get_behavior_vector(hd, cluster_users[c]) for c in target_clusters}
    mean_mats = {c: get_mean_matrix(hd, cluster_users[c]) for c in target_clusters}

    n_target = len(target_clusters)
    fig = plt.figure(figsize=(6 * n_target, 10))
    fig.suptitle("Suspicious vs Human Reference Clusters\n(Top: behavior vector | Bottom: mean hour×year matrix)",
                 fontsize=13)
    gs = gridspec.GridSpec(2, n_target, figure=fig, hspace=0.4, wspace=0.3)

    for col_idx, c in enumerate(target_clusters):
        row_stats = stats[stats["cluster"] == c].iloc[0]
        label = f"Cluster {c}\nn={int(row_stats['count']):,}  Entropy={row_stats['entropy']:.2f}  peak={int(row_stats['peak_hour']):02d}h"
        is_suspicious = c in SUSPICIOUS

        # Top: behavior vector bar
        ax_top = fig.add_subplot(gs[0, col_idx])
        bv = bvectors[c]
        bar_colors = ["#DD8452" if h in MORNING_HOURS else "#4C72B0" for h in range(24)]
        ax_top.bar(range(24), bv, color=bar_colors, alpha=0.85)
        ax_top.set_title(label, fontsize=9, color="#DD8452" if is_suspicious else "black")
        ax_top.set_xticks(range(0, 24, 4))
        ax_top.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=7)
        ax_top.set_xlabel("Hour", fontsize=8)
        ax_top.set_ylabel("Normalized probability", fontsize=8)
        ax_top.axvspan(5.5, 11.5, alpha=0.08, color="red", label="06-11h window")
        if col_idx == 0:
            ax_top.legend(fontsize=7)

        # Bottom: mean matrix heatmap
        ax_bot = fig.add_subplot(gs[1, col_idx])
        mat = mean_mats[c]
        vmax = np.nanpercentile(mat, 98) if not np.all(np.isnan(mat)) else 1
        im = ax_bot.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=vmax)
        ax_bot.set_title("Mean hour×year matrix", fontsize=8)
        ax_bot.set_xlabel("Hour", fontsize=8)
        ax_bot.set_ylabel("Year (0=oldest)", fontsize=8)
        ax_bot.set_xticks(range(0, 24, 4))
        ax_bot.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=7)
        ax_bot.set_yticks(range(0, N_YEARS, 3))
        ax_bot.set_yticklabels(range(0, N_YEARS, 3), fontsize=7)
        plt.colorbar(im, ax=ax_bot, fraction=0.03, pad=0.02)

    p = output_dir / "suspicious_vs_human_clusters.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {p}")

    # ── Plot 3: Individual user sample matrices from Cluster 10 ───────────────
    logger.info("[Sampling] Individual users from Cluster 10")
    N_SAMPLE = 9
    samples = sample_user_matrices(hd, cluster_users[10], n=N_SAMPLE)

    n_cols = 3
    n_rows = (len(samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    fig.suptitle(f"Cluster 10 — Individual User Hour×Year Matrices (n={len(samples)} samples)\n"
                 f"entropy=4.34  peak=05h  count=23,269", fontsize=11)

    for i, (uid, mat) in enumerate(samples):
        ax = axes[i // n_cols][i % n_cols]
        vmax_s = np.nanpercentile(mat, 98) if not np.all(np.isnan(mat)) else 1
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=vmax_s)
        ax.set_title(f"{str(uid)[:12]}…", fontsize=8)
        ax.set_xlabel("Hour", fontsize=7)
        ax.set_ylabel("Year", fontsize=7)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=6)
        ax.set_yticks(range(0, N_YEARS, 3))
        ax.set_yticklabels(range(0, N_YEARS, 3), fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    for i in range(len(samples), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    plt.tight_layout()
    p = output_dir / "cluster10_individual_samples.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {p}")

    logger.info("[Done]")


if __name__ == "__main__":
    main()
