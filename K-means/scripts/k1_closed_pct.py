"""
K1 Tier-2 Business Hours Validation
=====================================
For each K1 Tier-2 user, compute closed_pct:
    closed_pct = fraction of reviews posted during business closed hours

This signal is timezone-agnostic: if you post at 08:00 and the business
opens at 11:00, it's suspicious regardless of which timezone you're in.

Reference from cluster-level analysis (Sub3/Sub4 vs normal):
  - Normal clusters:   ~43% of reviews when businesses are open  (closed_pct ≈ 0.57)
  - C10-Sub3/Sub4:    ~9–14% of reviews when businesses are open (closed_pct ≈ 0.86–0.91)

Outputs:
  K-means/result/k1_closed_pct.parquet  — per-user closed_pct
  K-means/analysis/k1_closed_pct_dist.png
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from loguru import logger

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
K1_DIR       = Path(__file__).parents[1]
LT_FINAL_DIR = K1_DIR.parent / "lifetime kmeans" / "result" / "final"
ANALYSIS_OUT = K1_DIR / "analysis"
RESULT_OUT   = K1_DIR / "result"

# DOW mapping: DuckDB EXTRACT(DOW) → 0=Sun, 1=Mon, ..., 6=Sat
DAY_COLS = {
    0: ("Sunday_start_time",    "Sunday_end_time"),
    1: ("Monday_start_time",    "Monday_end_time"),
    2: ("Tuesday_start_time",   "Tuesday_end_time"),
    3: ("Wednesday_start_time", "Wednesday_end_time"),
    4: ("Thursday_start_time",  "Thursday_end_time"),
    5: ("Friday_start_time",    "Friday_end_time"),
    6: ("Saturday_start_time",  "Saturday_end_time"),
}


def time_to_min(t) -> int | None:
    """Convert a datetime.time (or None) to minutes since midnight."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    if hasattr(t, "hour"):
        return t.hour * 60 + t.minute
    return None


def build_hours_lookup(hours_df: pd.DataFrame) -> dict:
    """
    Returns dict: business_id → {dow: (open_min, close_min)} or {}
    NULL for a day = closed that day.
    close_min = 0 is treated as 1440 (midnight).
    """
    lookup = {}
    for row in hours_df.itertuples(index=False):
        bid = row.business_id
        lookup[bid] = {}
        for dow, (sc, ec) in DAY_COLS.items():
            s = time_to_min(getattr(row, sc))
            e = time_to_min(getattr(row, ec))
            if s is not None and e is not None:
                if e == 0:
                    e = 1440  # closes at midnight
                lookup[bid][dow] = (s, e)
    return lookup


def compute_closed_pct(reviews: pd.DataFrame, hours_lookup: dict) -> pd.DataFrame:
    """
    reviews: columns [user_id, business_id, review_hour, review_minute, day_of_week]
    Returns DataFrame [user_id, total_reviews, closed_reviews, closed_pct]
    """
    statuses = []
    for row in reviews.itertuples(index=False):
        bid = row.business_id
        dow = int(row.day_of_week)
        r_min = int(row.review_hour) * 60 + int(row.review_minute)

        if bid not in hours_lookup:
            statuses.append(None)   # no hours data → skip
            continue

        day_hours = hours_lookup[bid].get(dow)
        if day_hours is None:
            statuses.append(True)   # NULL hours = closed that day
        else:
            open_min, close_min = day_hours
            statuses.append(not (open_min <= r_min < close_min))

    reviews = reviews.copy()
    reviews["is_closed"] = statuses

    valid = reviews[reviews["is_closed"].notna()].copy()
    valid["is_closed"] = valid["is_closed"].astype(bool)

    result = (
        valid.groupby("user_id")
        .agg(total_reviews=("is_closed", "count"),
             closed_reviews=("is_closed", "sum"))
        .reset_index()
    )
    result["closed_pct"] = result["closed_reviews"] / result["total_reviews"]
    return result


def plot_distribution(cp_df: pd.DataFrame, threshold: float, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "K1 Tier-2 Users — closed_pct Distribution\n"
        "(fraction of reviews posted during business closed hours)",
        fontsize=11, y=1.01
    )

    ax = axes[0]
    ax.hist(cp_df["closed_pct"], bins=50, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(threshold, color="red", ls="--", lw=1.5, label=f"Threshold = {threshold:.2f}")
    ax.axvline(0.57, color="gray", ls=":", lw=1.2, label="Normal cluster baseline ≈ 0.57")
    ax.axvline(0.88, color="orange", ls=":", lw=1.2, label="Sub3/Sub4 baseline ≈ 0.88")
    flagged = (cp_df["closed_pct"] >= threshold).sum()
    ax.set_xlabel("closed_pct", fontsize=10)
    ax.set_ylabel("# users", fontsize=10)
    ax.set_title(
        f"n={len(cp_df):,} users with hours data\n"
        f"Flagged (≥{threshold:.2f}): {flagged:,}",
        fontsize=9
    )
    ax.legend(fontsize=8)

    ax2 = axes[1]
    percentiles = np.arange(0, 101, 5)
    pct_values  = np.percentile(cp_df["closed_pct"], percentiles)
    ax2.plot(percentiles, pct_values, marker="o", ms=4, color="#4C72B0")
    ax2.axhline(threshold, color="red", ls="--", lw=1.2, label=f"Threshold {threshold:.2f}")
    ax2.set_xlabel("Percentile", fontsize=10)
    ax2.set_ylabel("closed_pct", fontsize=10)
    ax2.set_title("Percentile curve", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {out_path.name}")


def main():
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    RESULT_OUT.mkdir(parents=True, exist_ok=True)

    # Load K1 Tier-2 users
    k1_all    = pd.read_parquet(RESULT_OUT / "anomaly_candidates.parquet")
    lifetime  = pd.read_parquet(LT_FINAL_DIR / "removal_final.parquet")
    tier2_ids = set(k1_all["user_id"].tolist()) - set(lifetime["user_id"].tolist())
    logger.info(f"[K1 Tier-2] {len(tier2_ids):,} users")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    ids_sql = ", ".join(str(i) for i in tier2_ids)
    reviews = con.execute(f"""
        SELECT
            r.user_id,
            r.business_id,
            CAST(EXTRACT(HOUR   FROM r.date) AS INTEGER) AS review_hour,
            CAST(EXTRACT(MINUTE FROM r.date) AS INTEGER) AS review_minute,
            CAST(EXTRACT(DOW    FROM r.date) AS INTEGER) AS day_of_week
        FROM review r
        WHERE r.user_id IN ({ids_sql})
    """).fetchdf()
    logger.info(f"[Reviews] {len(reviews):,} rows for {reviews['user_id'].nunique():,} users")

    biz_ids  = set(reviews["business_id"].unique())
    bids_sql = ", ".join(str(b) for b in biz_ids)
    hours_df = con.execute(f"""
        SELECT * FROM business_hours WHERE business_id IN ({bids_sql})
    """).fetchdf()
    con.close()
    logger.info(f"[Hours] {len(hours_df):,} businesses with hours data "
                f"(out of {len(biz_ids):,} reviewed)")

    hours_lookup = build_hours_lookup(hours_df)
    cp_df = compute_closed_pct(reviews, hours_lookup)
    logger.info(
        f"[closed_pct] {len(cp_df):,} users with enough hours data\n"
        f"  mean={cp_df['closed_pct'].mean():.3f}  "
        f"  median={cp_df['closed_pct'].median():.3f}  "
        f"  p75={cp_df['closed_pct'].quantile(0.75):.3f}  "
        f"  p90={cp_df['closed_pct'].quantile(0.90):.3f}"
    )

    # Distribution at various thresholds
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        n = (cp_df["closed_pct"] >= thr).sum()
        logger.info(f"  closed_pct >= {thr:.2f}: {n:,} users ({n/len(cp_df)*100:.1f}%)")

    THRESHOLD = 0.75
    plot_distribution(cp_df, THRESHOLD, ANALYSIS_OUT / "k1_closed_pct_dist.png")

    cp_df.to_parquet(RESULT_OUT / "k1_closed_pct.parquet", index=False)
    logger.info(f"[Saved] k1_closed_pct.parquet  ({len(cp_df):,} users)")
    logger.info("[Done]")


if __name__ == "__main__":
    main()
