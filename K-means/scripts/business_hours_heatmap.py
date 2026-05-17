"""
For each sub-cluster within C10 and C19:
  Compute the average business opening probability at each (day, hour) cell
  across all businesses reviewed by users in that sub-cluster.
  → 7-day × 24-hour heatmap per sub-cluster.

If heatmaps look similar across sub-clusters, then Sub3/Sub4's morning activity
cannot be explained by "special 24-hour shops" — it's genuine posting-outside-hours anomaly.
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from loguru import logger

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
sub_path    = Path(__file__).parents[1] / "analysis" / "subclustering" / "subclustering_assignments.parquet"
output_dir  = Path(__file__).parents[1] / "analysis" / "deep_profile"

PARENT_CLUSTERS = [10, 19]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DAY_COLS = [
    ("Monday_start_time",    "Monday_end_time"),
    ("Tuesday_start_time",   "Tuesday_end_time"),
    ("Wednesday_start_time", "Wednesday_end_time"),
    ("Thursday_start_time",  "Thursday_end_time"),
    ("Friday_start_time",    "Friday_end_time"),
    ("Saturday_start_time",  "Saturday_end_time"),
    ("Sunday_start_time",    "Sunday_end_time"),
]


def build_open_matrix(hours_row: pd.Series) -> np.ndarray:
    """
    Given one row from business_hours, return a (7, 24) bool array
    where True = business is open at that (day, hour).
    """
    mat = np.zeros((7, 24), dtype=np.float32)
    for d, (sc, ec) in enumerate(DAY_COLS):
        start = hours_row[sc]
        end   = hours_row[ec]
        if pd.isna(start) or pd.isna(end):
            mat[d, :] = np.nan
            continue
        sh, eh = start.hour, end.hour
        if sh <= eh:
            mat[d, sh:eh + 1] = 1.0
        else:   # overnight: e.g. 22:00 – 02:00
            mat[d, sh:] = 1.0
            mat[d, :eh + 1] = 1.0
    return mat


def avg_open_matrix(con, biz_ids: list) -> np.ndarray:
    """Mean (7, 24) open probability across all businesses with hours data."""
    if not biz_ids:
        return np.full((7, 24), np.nan)

    ids_sql = ", ".join(str(b) for b in biz_ids)
    hours_df = con.execute(
        f"SELECT * FROM business_hours WHERE business_id IN ({ids_sql})"
    ).fetchdf()

    if hours_df.empty:
        return np.full((7, 24), np.nan)

    mats = []
    for _, row in hours_df.iterrows():
        m = build_open_matrix(row)
        if not np.all(np.isnan(m)):
            mats.append(m)

    if not mats:
        return np.full((7, 24), np.nan)

    return np.nanmean(np.stack(mats, axis=0), axis=0)   # (7, 24)


def fetch_reviewed_biz(con, int_ids: list) -> list:
    ids_sql = ", ".join(str(i) for i in int_ids)
    df = con.execute(
        f"SELECT DISTINCT business_id FROM review WHERE user_id IN ({ids_sql})"
    ).fetchdf()
    return df["business_id"].tolist()


def plot_heatmaps(parent: int, sub_data: list):
    """
    sub_data: list of (label, n_users, morning_pct, open_mat_7x24)
              sorted by morning_pct descending
    """
    n = len(sub_data)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.5))
    fig.suptitle(
        f"Cluster {parent} — Mean Business Operating Hours per Sub-cluster\n"
        f"(X=7 days, Y=24 hours | color = % of reviewed businesses open at that slot)\n"
        f"Sorted by morning activity %, high → low",
        fontsize=12
    )

    vmin, vmax = 0.0, 100.0

    for idx, (label, n_users, morning_pct, mat) in enumerate(sub_data):
        ax = axes[idx // n_cols][idx % n_cols]
        im = ax.imshow(mat * 100, aspect="auto", cmap="YlGn",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{label}  n={n_users:,}\nmorning={morning_pct:.1f}%",
                     fontsize=8)
        ax.set_xlabel("Day of week", fontsize=7)
        ax.set_ylabel("Hour", fontsize=7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(DAYS, fontsize=7)
        ax.set_yticks(range(0, 24, 4))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 4)], fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format="%.0f%%")

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    p = output_dir / f"cluster{parent}_biz_hours_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {p}")


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    sub_df = pd.read_parquet(sub_path)
    con    = duckdb.connect(str(DB_PATH), read_only=True)

    for parent in PARENT_CLUSTERS:
        logger.info(f"[Start] Cluster {parent}")
        sub_clusters = sorted(
            sub_df[sub_df["parent_cluster"] == parent]["sub_cluster"].unique()
        )

        # Compute morning ratio per sub to sort
        sub_data = []
        for sc in sub_clusters:
            ids = sub_df[
                (sub_df["parent_cluster"] == parent) &
                (sub_df["sub_cluster"] == sc)
            ]["user_id"].tolist()

            biz_ids = fetch_reviewed_biz(con, ids)
            mat     = avg_open_matrix(con, biz_ids)

            # Morning ratio from review timestamps
            ids_sql = ", ".join(str(i) for i in ids)
            rev_hours = con.execute(
                f"SELECT EXTRACT(HOUR FROM date) AS h FROM review WHERE user_id IN ({ids_sql})"
            ).fetchdf()["h"]
            morning_pct = (rev_hours.isin(range(6, 12))).mean() * 100

            label = f"Sub{sc}"
            logger.info(
                f"  Sub{sc}: {len(ids):,} users | "
                f"{len(biz_ids):,} unique biz | morning={morning_pct:.1f}%"
            )
            sub_data.append((label, len(ids), morning_pct, mat))

        # Sort by morning_pct descending
        sub_data.sort(key=lambda x: x[2], reverse=True)
        plot_heatmaps(parent, sub_data)
        logger.info(f"[Done] Cluster {parent}")

    con.close()
    logger.info("[All done]")


if __name__ == "__main__":
    main()
