import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from loguru import logger

HOUR_COLS = [f"h{h:02d}" for h in range(24)]
N_YEARS = 18
N_CLUSTERS = 20

matrix_path = Path(__file__).parents[2] / "user behavier matrix" / "result" / "user_hour_matrix.parquet"
assign_path = Path(__file__).parents[1] / "analysis" / "kmeans_clustering" / "kmeans_assignments.parquet"
stats_path  = Path(__file__).parents[1] / "analysis" / "kmeans_clustering" / "cluster_stats.csv"
output_path = Path(__file__).parents[1] / "analysis" / "kmeans_clustering" / "cluster_matrix_overview.png"


def main():
    logger.info("[Loading] Assignments & stats")
    assignments = pd.read_parquet(assign_path)          # user_id, cluster
    stats = pd.read_csv(stats_path)                     # cluster, count, entropy, peak_hour
    stats = stats.sort_values("entropy").reset_index(drop=True)
    cluster_order = stats["cluster"].tolist()

    logger.info("[Loading] Matrix")
    matrix = pd.read_parquet(matrix_path)
    logger.info(f"[Success] Matrix: {matrix.shape}")

    # Keep only valid users (those in assignments)
    valid_ids = set(assignments["user_id"])
    mask = matrix.index.get_level_values("user_id").isin(valid_ids)
    matrix = matrix[mask]

    # Replace -1 (pre-account) with nan
    hour_data = matrix[HOUR_COLS].copy().astype(float)
    hour_data[hour_data == -1] = np.nan

    # Add cluster label via user_id level
    user_ids = matrix.index.get_level_values("user_id")
    uid_to_cluster = assignments.set_index("user_id")["cluster"]
    cluster_col = uid_to_cluster.reindex(user_ids).values
    hour_data["cluster"] = cluster_col
    hour_data["year_idx"] = matrix.index.get_level_values("year")

    logger.info("[Computing] Mean matrix per cluster (this may take a minute)")

    # Compute per-cluster mean: group by (cluster, year_idx) → mean over users
    grp = hour_data.groupby(["cluster", "year_idx"])[HOUR_COLS].mean()

    logger.info("[Plotting] Building overview figure")

    n_cols = 5
    n_rows = (N_CLUSTERS + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    fig.suptitle(
        "K-Means Cluster Mean Hour×Year Matrix (sorted by entropy, low→high)",
        fontsize=14, y=1.01
    )

    for plot_idx, cluster_id in enumerate(cluster_order):
        ax = axes[plot_idx // n_cols][plot_idx % n_cols]

        row = stats[stats["cluster"] == cluster_id].iloc[0]
        ent = row["entropy"]
        count = int(row["count"])
        peak = int(row["peak_hour"])

        # Extract mean matrix for this cluster: shape (N_YEARS, 24)
        try:
            sub = grp.loc[cluster_id].sort_index()   # indexed by actual year (e.g. 2005-2022)
            all_years = sub.index.tolist()
            year_min = min(all_years)
            sub = sub.reindex(range(year_min, year_min + N_YEARS))
            mat = sub[HOUR_COLS].values          # (18, 24)
        except KeyError:
            mat = np.full((N_YEARS, 24), np.nan)

        vmax = np.nanpercentile(mat, 98) if not np.all(np.isnan(mat)) else 1
        im = ax.imshow(
            mat, aspect="auto", cmap="YlOrRd",
            interpolation="nearest", vmin=0, vmax=vmax
        )
        ax.set_title(
            f"Cluster {cluster_id}  n={count:,}\nEntropy={ent:.2f}  peak={peak:02d}h",
            fontsize=8
        )
        ax.set_xlabel("Hour", fontsize=7)
        ax.set_ylabel("Year (0=oldest)", fontsize=7)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=6)
        ax.set_yticks(range(0, N_YEARS, 3))
        ax.set_yticklabels(range(0, N_YEARS, 3), fontsize=6)

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Hide unused axes
    for idx in range(N_CLUSTERS, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
