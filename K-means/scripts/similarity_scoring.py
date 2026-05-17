"""
Route A: Cosine similarity scoring against Sub3/Sub4 anomaly centroids.
Applies to all 287k valid users (MIN_REVIEWS >= 5).
Outputs:
  - result/all_scores.parquet      : every user with similarity score + cluster info
  - result/anomaly_candidates.parquet : users above threshold
  - Plots: score distribution, threshold visualisation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

HOUR_COLS    = [f"h{h:02d}" for h in range(24)]
MIN_REVIEWS  = 5

matrix_path  = Path(__file__).parents[2] / "user behavier matrix" / "result" / "user_hour_matrix.parquet"
sub_path     = Path(__file__).parents[1] / "analysis" / "subclustering" / "subclustering_assignments.parquet"
kmeans_path  = Path(__file__).parents[1] / "analysis" / "kmeans_clustering" / "kmeans_assignments.parquet"
result_dir   = Path(__file__).parents[1] / "result"
output_dir   = Path(__file__).parents[1] / "analysis" / "similarity_scoring"


# ── behavior vector computation (same algorithm as behavior_kmeans.py) ────────

def compute_behavior_vectors(matrix: pd.DataFrame) -> tuple[np.ndarray, list]:
    eps = 1e-10
    hour_data = matrix[HOUR_COLS].copy().astype(float)
    hour_data[hour_data == -1] = np.nan

    total_reviews = hour_data.groupby(level="user_id").sum(min_count=1).sum(axis=1)
    valid_users   = total_reviews[total_reviews >= MIN_REVIEWS].index.tolist()
    logger.info(f"[Filter] Valid users (>= {MIN_REVIEWS} reviews): {len(valid_users):,}")

    mask     = hour_data.index.get_level_values("user_id").isin(set(valid_users))
    filtered = hour_data[mask]

    row_totals = filtered.sum(axis=1)
    active     = row_totals > 0
    normalized = filtered.copy()
    normalized[active]  = filtered[active].div(row_totals[active], axis=0)
    normalized[~active] = 0.0
    normalized = normalized.fillna(0.0)

    accumulated  = normalized.groupby(level="user_id").sum()
    user_totals  = accumulated.sum(axis=1)
    bvec         = accumulated.div(user_totals + eps, axis=0)

    return bvec.values.astype(np.float32), bvec.index.tolist()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    result_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("[Loading] Matrix")
    matrix   = pd.read_parquet(matrix_path)
    sub_df   = pd.read_parquet(sub_path)
    kmeans_df = pd.read_parquet(kmeans_path)
    logger.info(f"[Success] Matrix: {matrix.shape}")

    # Compute behavior vectors for all 287k users
    logger.info("[Computing] Behavior vectors for all valid users")
    vectors, user_ids = compute_behavior_vectors(matrix)
    logger.info(f"[Done] Vectors shape: {vectors.shape}")

    uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    # Build anomaly centroid from Sub3 + Sub4 of Cluster 10
    anomaly_ids = set(
        sub_df[
            (sub_df["parent_cluster"] == 10) &
            (sub_df["sub_cluster"].isin([3, 4]))
        ]["user_id"].tolist()
    )
    anomaly_idx = [uid_to_idx[uid] for uid in anomaly_ids if uid in uid_to_idx]
    anomaly_vecs = vectors[anomaly_idx]
    centroid = anomaly_vecs.mean(axis=0, keepdims=True)   # (1, 24)
    logger.info(f"[Centroid] Built from {len(anomaly_idx):,} Sub3+Sub4 users")

    # Cosine similarity: all 287k vs centroid
    logger.info("[Scoring] Computing cosine similarity for all users")
    scores = cosine_similarity(vectors, centroid).flatten()   # (287k,)
    logger.info(f"[Scores] min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")

    # Scores for known anomalies
    anomaly_scores = scores[anomaly_idx]
    logger.info(
        f"[Anomaly ref] Sub3+Sub4 scores: "
        f"p5={np.percentile(anomaly_scores, 5):.4f}  "
        f"p10={np.percentile(anomaly_scores, 10):.4f}  "
        f"median={np.median(anomaly_scores):.4f}"
    )

    # Threshold = 5th percentile of known anomaly scores
    # (anyone at least as similar as the least-similar known anomaly, conservatively)
    threshold = float(np.percentile(anomaly_scores, 5))
    logger.info(f"[Threshold] {threshold:.4f}  (5th pct of Sub3+Sub4 similarity)")

    # Build result dataframe
    result_df = pd.DataFrame({
        "user_id":    user_ids,
        "similarity": scores,
    })
    result_df = result_df.merge(
        kmeans_df[["user_id", "cluster"]], on="user_id", how="left"
    )
    # Flag known anomalies
    result_df["is_known_anomaly"] = result_df["user_id"].isin(anomaly_ids).astype(int)
    result_df = result_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    # Save full scores
    result_df.to_parquet(result_dir / "all_scores.parquet", index=False)
    logger.info(f"[Saved] all_scores.parquet  ({len(result_df):,} users)")

    # Anomaly candidates
    candidates = result_df[result_df["similarity"] >= threshold].copy()
    candidates.to_parquet(result_dir / "anomaly_candidates.parquet", index=False)

    total_users   = len(result_df)
    n_candidates  = len(candidates)
    pct           = n_candidates / total_users * 100
    logger.info(
        f"\n{'='*55}\n"
        f"  Total valid users   : {total_users:>10,}\n"
        f"  Anomaly candidates  : {n_candidates:>10,}\n"
        f"  Proportion          : {pct:>9.3f}%\n"
        f"  Threshold           : {threshold:>9.4f}\n"
        f"{'='*55}"
    )

    # ── Plot: score distribution ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cosine Similarity to Sub3+Sub4 Anomaly Centroid", fontsize=13)

    # Left: full distribution
    ax = axes[0]
    ax.hist(scores, bins=100, color="#4C72B0", alpha=0.8, label="All 287k users")
    ax.hist(anomaly_scores, bins=30, color="#DD8452", alpha=0.85,
            label=f"Known anomaly (Sub3+Sub4, n={len(anomaly_idx):,})")
    ax.axvline(threshold, color="red", lw=1.5, ls="--",
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Cosine similarity", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Full Distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Right: zoomed high-similarity tail
    ax2 = axes[1]
    tail_scores = scores[scores >= threshold - 0.05]
    ax2.hist(tail_scores, bins=60, color="#4C72B0", alpha=0.8)
    ax2.hist(anomaly_scores, bins=30, color="#DD8452", alpha=0.85,
             label=f"Sub3+Sub4")
    ax2.axvline(threshold, color="red", lw=1.5, ls="--",
                label=f"Threshold = {threshold:.3f}")
    ax2.set_xlabel("Cosine similarity", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title(f"Tail View (≥ {threshold - 0.05:.2f})  |  candidates = {n_candidates:,}  ({pct:.2f}%)",
                  fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    p = output_dir / "similarity_distribution.png"
    plt.savefig(p, dpi=150)
    plt.close()
    logger.info(f"[Saved] {p}")

    logger.info("[Done]")


if __name__ == "__main__":
    main()
