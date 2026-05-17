import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger
from attrs import define, field
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.stats import entropy as scipy_entropy

HOUR_COLS = [f"h{h:02d}" for h in range(24)]
N_YEARS = 18
N_HOURS = 24
MIN_REVIEWS = 5    # exclude sparse users
N_CLUSTERS = 20
SAMPLE_PER_CLUSTER = 3   # sample heatmaps per cluster


@define
class BehaviorKMeans:
    matrix_path: Path = field(
        init=False,
        default=Path(__file__).parents[2] / "user behavier matrix" / "result" / "user_hour_matrix.parquet"
    )
    output_path: Path = field(
        init=False,
        default=Path(__file__).parents[1] / "analysis" / "kmeans_clustering"
    )

    def load_matrix(self) -> pd.DataFrame:
        logger.info("[Loading] Matrix")
        matrix = pd.read_parquet(self.matrix_path)
        logger.info(f"[Success] Matrix: {matrix.shape}")
        return matrix

    def compute_behavior_vectors(self, matrix: pd.DataFrame) -> tuple[np.ndarray, list]:
        """
        Per active year: normalize hour distribution → sum across years → re-normalize.
        Returns (n_valid_users, 24) array and corresponding user_id list.
        """
        logger.info("[Computing] Behavior vectors")
        eps = 1e-10

        hour_data = matrix[HOUR_COLS].copy()
        hour_data[hour_data == -1] = np.nan   # mask pre-account years

        # Total reviews per user (across all active years)
        total_reviews = hour_data.groupby(level="user_id").sum(min_count=1).sum(axis=1)
        valid_users = total_reviews[total_reviews >= MIN_REVIEWS].index.tolist()
        logger.info(f"[Filter] Users with >= {MIN_REVIEWS} reviews: {len(valid_users):,} / {total_reviews.shape[0]:,}")

        # Compute per-year normalized distributions
        mask = hour_data.index.get_level_values("user_id").isin(set(valid_users))
        filtered = hour_data[mask]

        # Per row (year): normalize
        row_totals = filtered.sum(axis=1)
        active_mask = row_totals > 0
        normalized = filtered.copy()
        normalized[active_mask] = filtered[active_mask].div(row_totals[active_mask], axis=0)
        normalized[~active_mask] = 0.0
        normalized = normalized.fillna(0.0)   # nan rows = -1 years → 0

        # Sum across years per user
        accumulated = normalized.groupby(level="user_id").sum()   # (n_users, 24)

        # Re-normalize per user
        user_totals = accumulated.sum(axis=1)
        behavior_vectors = accumulated.div(user_totals + eps, axis=0)

        valid_user_ids = behavior_vectors.index.tolist()
        vectors = behavior_vectors.values.astype(np.float32)

        logger.info(f"[Done] Behavior vectors shape: {vectors.shape}")
        return vectors, valid_user_ids

    def run_kmeans(self, vectors: np.ndarray) -> np.ndarray:
        logger.info(f"[KMeans] K={N_CLUSTERS} on {vectors.shape[0]:,} users")
        kmeans = MiniBatchKMeans(
            n_clusters=N_CLUSTERS,
            random_state=42,
            batch_size=65536,
            n_init=5,
            verbose=0,
        )
        labels = kmeans.fit_predict(vectors)
        logger.info("[KMeans] Done")
        return labels, kmeans.cluster_centers_

    def plot_cluster_overview(self, centroids: np.ndarray, cluster_stats: pd.DataFrame):
        """Grid of all cluster centroids sorted by peak hour."""
        order = np.argsort(centroids.argmax(axis=1))
        n_cols = 5
        n_rows = (N_CLUSTERS + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
        fig.suptitle(f"K-Means Cluster Centroids (K={N_CLUSTERS}, min_reviews={MIN_REVIEWS})", fontsize=13)

        for idx, cluster_id in enumerate(order):
            ax = axes[idx // n_cols][idx % n_cols]
            centroid = centroids[cluster_id]
            peak_hour = centroid.argmax()
            ent = -np.sum(centroid * np.log2(centroid + 1e-10))
            n_users = cluster_stats.loc[cluster_stats["cluster"] == cluster_id, "count"].values[0]

            ax.bar(range(N_HOURS), centroid, color="#4C72B0", alpha=0.85)
            ax.axvline(peak_hour, color="red", lw=1, ls="--", alpha=0.6)
            ax.set_title(
                f"Cluster {cluster_id}\nn={n_users:,} | peak={peak_hour:02d}h | Entropy={ent:.2f}",
                fontsize=8
            )
            ax.set_xticks(range(0, N_HOURS, 6))
            ax.set_xticklabels([f"{h:02d}" for h in range(0, N_HOURS, 6)], fontsize=7)
            ax.set_ylim(0, centroid.max() * 1.4)
            ax.tick_params(axis='y', labelsize=6)

        # Hide unused axes
        for idx in range(N_CLUSTERS, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        plt.tight_layout()
        save_path = self.output_path / "cluster_overview.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"[Saved] {save_path}")

    def plot_cluster_heatmaps(
        self,
        cluster_id: int,
        centroid: np.ndarray,
        sample_user_ids: list,
        matrix: pd.DataFrame,
    ):
        """For one cluster: centroid bar + sample raw heatmaps."""
        n_samples = len(sample_user_ids)
        fig = plt.figure(figsize=(5 * (n_samples + 1), 4))
        gs = gridspec.GridSpec(1, n_samples + 1, figure=fig, wspace=0.3)

        ent = -np.sum(centroid * np.log2(centroid + 1e-10))

        # Centroid
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.bar(range(N_HOURS), centroid, color="#4C72B0", alpha=0.85)
        ax0.set_title(f"Cluster {cluster_id} centroid\nEntropy={ent:.2f}", fontsize=9)
        ax0.set_xlabel("Hour")
        ax0.set_xticks(range(0, N_HOURS, 4))

        # Sample heatmaps
        import seaborn as sns
        for i, uid in enumerate(sample_user_ids):
            ax = fig.add_subplot(gs[0, i + 1])
            try:
                user_data = matrix.loc[uid][HOUR_COLS].values.astype(float)
                user_data = np.where(user_data == -1, 0, user_data)  # (18, 24)
                sns.heatmap(user_data, ax=ax, cmap="Blues", cbar=False, xticklabels=4)
                ax.set_title(f"{str(uid)[:10]}…", fontsize=8)
                ax.set_xlabel("Hour")
                ax.set_ylabel("Year")
            except Exception:
                ax.set_visible(False)

        plt.suptitle(f"Cluster {cluster_id} — Centroid + Sample Users", fontsize=10)
        plt.tight_layout()
        save_path = self.output_path / f"cluster_{cluster_id:02d}_heatmap.png"
        plt.savefig(save_path, dpi=130)
        plt.close()

    def main(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

        matrix = self.load_matrix()
        vectors, valid_user_ids = self.compute_behavior_vectors(matrix)

        labels, centroids = self.run_kmeans(vectors)

        # Save assignments
        result_df = pd.DataFrame({
            "user_id": valid_user_ids,
            "cluster": labels,
        })
        result_df.to_parquet(self.output_path / "kmeans_assignments.parquet", index=False)
        logger.info(f"[Saved] kmeans_assignments.parquet | Rows: {len(result_df):,}")

        # Cluster stats
        cluster_stats = result_df.groupby("cluster").size().reset_index(name="count")
        cluster_stats["entropy"] = [
            -np.sum(centroids[c] * np.log2(centroids[c] + 1e-10))
            for c in cluster_stats["cluster"]
        ]
        cluster_stats["peak_hour"] = centroids[cluster_stats["cluster"].values].argmax(axis=1)
        cluster_stats = cluster_stats.sort_values("entropy")
        cluster_stats.to_csv(self.output_path / "cluster_stats.csv", index=False)

        logger.info(f"\n[Cluster Stats]\n{cluster_stats.to_string(index=False)}")

        # Overview plot
        self.plot_cluster_overview(centroids, cluster_stats)

        # Per-cluster heatmaps (sample users)
        logger.info("[Heatmaps] Generating per-cluster sample heatmaps")
        for cluster_id in range(N_CLUSTERS):
            cluster_users = result_df[result_df["cluster"] == cluster_id]["user_id"].tolist()
            sample = cluster_users[:SAMPLE_PER_CLUSTER]
            self.plot_cluster_heatmaps(cluster_id, centroids[cluster_id], sample, matrix)

        logger.info("[Done] All complete")


if __name__ == "__main__":
    clustering = BehaviorKMeans()
    clustering.main()
