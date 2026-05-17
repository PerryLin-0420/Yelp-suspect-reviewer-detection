import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger
from sklearn.cluster import MiniBatchKMeans

HOUR_COLS = [f"h{h:02d}" for h in range(24)]
N_YEARS = 18
N_SUB_CLUSTERS = 10     # sub-clusters per suspicious cluster
MORNING_HOURS = list(range(6, 12))
SAMPLE_PER_SUBCLUSTER = 4

matrix_path = Path(__file__).parents[2] / "user behavier matrix" / "result" / "user_hour_matrix.parquet"
assign_path = Path(__file__).parents[1] / "analysis" / "kmeans_clustering" / "kmeans_assignments.parquet"
output_dir  = Path(__file__).parents[1] / "analysis" / "subclustering"

TARGET_CLUSTERS = [10, 19]


def compute_behavior_vectors(hour_data: pd.DataFrame, user_ids: list) -> tuple[np.ndarray, list]:
    eps = 1e-10
    mask = hour_data.index.get_level_values("user_id").isin(set(user_ids))
    filtered = hour_data[mask].copy().astype(float)
    filtered[filtered == -1] = np.nan

    row_totals = filtered.sum(axis=1)
    active = row_totals > 0
    normalized = filtered.copy()
    normalized[active] = filtered[active].div(row_totals[active], axis=0)
    normalized[~active] = 0.0
    normalized = normalized.fillna(0.0)

    accumulated = normalized.groupby(level="user_id").sum()
    user_totals = accumulated.sum(axis=1)
    bvec = accumulated.div(user_totals + eps, axis=0)

    return bvec.values.astype(np.float32), bvec.index.tolist()


def get_mean_matrix(hour_data: pd.DataFrame, user_ids: list) -> np.ndarray:
    mask = hour_data.index.get_level_values("user_id").isin(set(user_ids))
    sub = hour_data[mask].copy().astype(float)
    sub[sub == -1] = np.nan
    grp = sub.groupby(level="year")[HOUR_COLS].mean().sort_index()
    all_years = grp.index.tolist()
    if not all_years:
        return np.full((N_YEARS, 24), np.nan)
    year_min = min(all_years)
    return grp.reindex(range(year_min, year_min + N_YEARS)).values


def plot_subcluster_overview(
    parent_cluster: int,
    sub_labels: np.ndarray,
    vectors: np.ndarray,
    user_ids: list,
    hour_data: pd.DataFrame,
):
    """Grid: each sub-cluster shows behavior vector (top) + mean matrix (bottom)."""
    unique = sorted(set(sub_labels))
    n = len(unique)
    morning_cols = [f"h{h:02d}" for h in MORNING_HOURS]

    fig = plt.figure(figsize=(5 * n, 10))
    fig.suptitle(
        f"Cluster {parent_cluster} → {n} sub-clusters (sorted by morning ratio, high→low)",
        fontsize=13
    )
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.3)

    # Compute morning ratio per sub-cluster for sorting
    uid_arr = np.array(user_ids)
    sub_morning = {}
    for sc in unique:
        sc_ids = uid_arr[sub_labels == sc].tolist()
        sc_vecs = vectors[sub_labels == sc]
        mr = sc_vecs[:, MORNING_HOURS].sum(axis=1).mean()
        sub_morning[sc] = (mr, sc_ids, sc_vecs)

    sorted_subs = sorted(unique, key=lambda sc: sub_morning[sc][0], reverse=True)

    for col_idx, sc in enumerate(sorted_subs):
        mr, sc_ids, sc_vecs = sub_morning[sc]
        centroid = sc_vecs.mean(axis=0)
        ent = -np.sum(centroid * np.log2(centroid + 1e-10))
        peak = centroid.argmax()
        n_users = len(sc_ids)

        label = f"Sub {sc}  n={n_users:,}\nEntropy={ent:.2f}  peak={peak:02d}h\nmorning={mr*100:.1f}%"

        # Top: behavior vector
        ax_top = fig.add_subplot(gs[0, col_idx])
        bar_colors = ["#DD8452" if h in MORNING_HOURS else "#4C72B0" for h in range(24)]
        ax_top.bar(range(24), centroid, color=bar_colors, alpha=0.85)
        ax_top.axvspan(5.5, 11.5, alpha=0.07, color="red")
        ax_top.set_title(label, fontsize=8)
        ax_top.set_xticks(range(0, 24, 4))
        ax_top.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=7)
        ax_top.set_xlabel("Hour", fontsize=7)

        # Bottom: mean matrix heatmap
        ax_bot = fig.add_subplot(gs[1, col_idx])
        mat = get_mean_matrix(hour_data, sc_ids)
        vmax = np.nanpercentile(mat, 98) if not np.all(np.isnan(mat)) else 1
        im = ax_bot.imshow(mat, aspect="auto", cmap="YlOrRd",
                           interpolation="nearest", vmin=0, vmax=vmax)
        ax_bot.set_xlabel("Hour", fontsize=7)
        ax_bot.set_ylabel("Year", fontsize=7)
        ax_bot.set_xticks(range(0, 24, 4))
        ax_bot.set_xticklabels([f"{h:02d}" for h in range(0, 24, 4)], fontsize=6)
        ax_bot.set_yticks(range(0, N_YEARS, 3))
        ax_bot.set_yticklabels(range(0, N_YEARS, 3), fontsize=6)
        plt.colorbar(im, ax=ax_bot, fraction=0.03, pad=0.02)

    plt.tight_layout()
    p = output_dir / f"cluster{parent_cluster}_subclusters.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {p}")


def main():
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[Loading] Matrix and assignments")
    assignments = pd.read_parquet(assign_path)
    matrix = pd.read_parquet(matrix_path)
    logger.info(f"[Success] Matrix: {matrix.shape}")

    hour_data = matrix[HOUR_COLS]
    all_sub_results = []

    for parent_cluster in TARGET_CLUSTERS:
        logger.info(f"[Start] Sub-clustering Cluster {parent_cluster}")
        cluster_ids = assignments[assignments["cluster"] == parent_cluster]["user_id"].tolist()
        logger.info(f"  Users: {len(cluster_ids):,}")

        vectors, valid_ids = compute_behavior_vectors(hour_data, cluster_ids)
        logger.info(f"  Behavior vectors: {vectors.shape}")

        kmeans = MiniBatchKMeans(
            n_clusters=N_SUB_CLUSTERS,
            random_state=42,
            batch_size=min(65536, len(valid_ids)),
            n_init=10,
        )
        sub_labels = kmeans.fit_predict(vectors)

        # Log sub-cluster stats
        for sc in sorted(set(sub_labels)):
            sc_vecs = vectors[sub_labels == sc]
            centroid = sc_vecs.mean(axis=0)
            mr = centroid[MORNING_HOURS].sum()
            ent = -np.sum(centroid * np.log2(centroid + 1e-10))
            logger.info(
                f"  Sub {sc}: n={( sub_labels==sc).sum():,}  "
                f"entropy={ent:.2f}  peak={centroid.argmax():02d}h  morning={mr*100:.1f}%"
            )

        plot_subcluster_overview(parent_cluster, sub_labels, vectors, valid_ids, hour_data)

        # Save sub-cluster assignments
        df = pd.DataFrame({
            "user_id": valid_ids,
            "parent_cluster": parent_cluster,
            "sub_cluster": sub_labels,
        })
        all_sub_results.append(df)

    combined = pd.concat(all_sub_results, ignore_index=True)
    save_path = output_dir / "subclustering_assignments.parquet"
    combined.to_parquet(save_path, index=False)
    logger.info(f"[Saved] {save_path} | Rows: {len(combined):,}")
    logger.info("[Done]")


if __name__ == "__main__":
    main()
