"""
Deep profile analysis for ALL sub-clusters within Cluster 10 & Cluster 19.
Each parent cluster gets its own figure comparing all 10 sub-clusters:
  1. Business hours match  — open / closed / no_hours (stacked bar)
  2. Photo proxy           — % users with compliment_photos > 0
  3. Text repetition       — % users with ≥1 duplicate review text
Sub-clusters sorted by morning ratio (high → most suspicious on left).
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger

DB_PATH     = Path("E:/Project/Yelp/database/YELP.duckdb")
sub_path    = Path(__file__).parents[1] / "analysis" / "subclustering" / "subclustering_assignments.parquet"
output_dir  = Path(__file__).parents[1] / "analysis" / "deep_profile"

PARENT_CLUSTERS = [10, 19]
MORNING_HOURS   = list(range(6, 12))

DAY_MAP = {
    0: ("Monday_start_time",    "Monday_end_time"),
    1: ("Tuesday_start_time",   "Tuesday_end_time"),
    2: ("Wednesday_start_time", "Wednesday_end_time"),
    3: ("Thursday_start_time",  "Thursday_end_time"),
    4: ("Friday_start_time",    "Friday_end_time"),
    5: ("Saturday_start_time",  "Saturday_end_time"),
    6: ("Sunday_start_time",    "Sunday_end_time"),
}


# ── data helpers ──────────────────────────────────────────────────────────────

def fetch_reviews(con, int_ids: list) -> pd.DataFrame:
    ids_sql = ", ".join(str(i) for i in int_ids)
    return con.execute(f"""
        SELECT r.user_id, r.business_id, r.text, r.date
        FROM review r WHERE r.user_id IN ({ids_sql})
    """).fetchdf()


def fetch_profiles(con, int_ids: list) -> pd.DataFrame:
    ids_sql = ", ".join(str(i) for i in int_ids)
    return con.execute(f"""
        SELECT user_id, compliment_photos FROM user WHERE user_id IN ({ids_sql})
    """).fetchdf()


def fetch_hours(con, biz_ids: list) -> pd.DataFrame:
    ids_sql = ", ".join(str(b) for b in biz_ids)
    return con.execute(
        f"SELECT * FROM business_hours WHERE business_id IN ({ids_sql})"
    ).fetchdf().set_index("business_id")


# ── metrics ───────────────────────────────────────────────────────────────────

def biz_hours_pct(reviews: pd.DataFrame, hours_df: pd.DataFrame) -> dict:
    results = []
    for _, row in reviews.iterrows():
        ts  = pd.Timestamp(row["date"])
        dow = ts.weekday()
        t   = ts.time()
        bid = row["business_id"]

        if bid not in hours_df.index:
            results.append("no_hours")
            continue

        start_col, end_col = DAY_MAP[dow]
        bh = hours_df.loc[bid]
        start, end = bh[start_col], bh[end_col]

        if pd.isna(start) or pd.isna(end):
            results.append("no_hours")
            continue

        is_open = (start <= end and start <= t <= end) or \
                  (start > end and (t >= start or t <= end))
        results.append("open" if is_open else "closed")

    if not results:
        return {"open": 0.0, "closed": 0.0, "no_hours": 100.0}
    ser = pd.Series(results).value_counts(normalize=True) * 100
    return {k: ser.get(k, 0.0) for k in ["open", "closed", "no_hours"]}


def dup_user_pct(reviews: pd.DataFrame) -> float:
    """% of users who have at least one duplicate review text."""
    if reviews.empty:
        return 0.0
    def _has_dup(s):
        texts = s.dropna().tolist()
        return len(texts) > len(set(texts))
    return reviews.groupby("user_id")["text"].apply(_has_dup).mean() * 100


def photo_has_pct(profiles: pd.DataFrame) -> float:
    """% of users with compliment_photos > 0."""
    if profiles.empty:
        return 0.0
    return (profiles["compliment_photos"] > 0).mean() * 100


def morning_ratio(reviews: pd.DataFrame) -> float:
    if reviews.empty:
        return 0.0
    hours = pd.to_datetime(reviews["date"]).dt.hour
    return (hours.isin(MORNING_HOURS)).mean() * 100


# ── per-parent analysis ───────────────────────────────────────────────────────

def analyze_parent(con, parent: int, sub_df: pd.DataFrame) -> pd.DataFrame:
    sub_ids_map = {
        sc: sub_df[(sub_df["parent_cluster"] == parent) &
                   (sub_df["sub_cluster"] == sc)]["user_id"].tolist()
        for sc in sorted(sub_df[sub_df["parent_cluster"] == parent]["sub_cluster"].unique())
    }

    # Pre-fetch all business hours for reviews in this parent cluster
    all_ids = [uid for ids in sub_ids_map.values() for uid in ids]
    all_reviews = fetch_reviews(con, all_ids)
    biz_ids = all_reviews["business_id"].unique().tolist()
    hours_df = fetch_hours(con, biz_ids)
    all_profiles = fetch_profiles(con, all_ids)

    rows = []
    for sc, ids in sub_ids_map.items():
        rev  = all_reviews[all_reviews["user_id"].isin(set(ids))]
        prof = all_profiles[all_profiles["user_id"].isin(set(ids))]

        bh   = biz_hours_pct(rev, hours_df)
        dup  = dup_user_pct(rev)
        photo = photo_has_pct(prof)
        mr   = morning_ratio(rev)

        rows.append({
            "sub_cluster": sc,
            "n_users":     len(ids),
            "n_reviews":   len(rev),
            "morning_pct": mr,
            "open_pct":    bh["open"],
            "closed_pct":  bh["closed"],
            "no_hours_pct": bh["no_hours"],
            "dup_user_pct": dup,
            "photo_pct":   photo,
        })
        logger.info(
            f"  [C{parent} Sub{sc}] n={len(ids):,} reviews={len(rev):,} "
            f"morning={mr:.1f}% open={bh['open']:.1f}% "
            f"dup={dup:.1f}% photo={photo:.1f}%"
        )

    return pd.DataFrame(rows).sort_values("morning_pct", ascending=False).reset_index(drop=True)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_parent(parent: int, df: pd.DataFrame):
    labels = [f"Sub{int(r.sub_cluster)}\nn={int(r.n_users):,}" for _, r in df.iterrows()]
    x = np.arange(len(df))

    fig = plt.figure(figsize=(max(16, len(df) * 1.6), 12))
    fig.suptitle(f"Cluster {parent} — All Sub-clusters Profile Analysis\n"
                 f"(sorted by morning activity %, high → low = most suspicious → least)",
                 fontsize=13)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── 1. Business hours stacked bar (full top row) ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    bottom = np.zeros(len(df))
    cat_specs = [
        ("open_pct",     "#4C72B0", "Open"),
        ("closed_pct",   "#DD8452", "Closed"),
        ("no_hours_pct", "#aaaaaa", "No hours data"),
    ]
    for col, cc, cl in cat_specs:
        vals = df[col].values
        ax1.bar(x, vals, bottom=bottom, color=cc, alpha=0.88, label=cl)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax1.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, fontweight="bold", color="white")
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("% of reviews", fontsize=10)
    ax1.set_title("Review Time vs Business Operating Hours", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(0, 115)
    # Mark morning % on secondary axis info
    for xi, row in enumerate(df.itertuples()):
        ax1.text(xi, 103, f"morning\n{row.morning_pct:.0f}%",
                 ha="center", va="bottom", fontsize=7, color="#AA3333")

    # ── 2. Open % line for clarity ───────────────────────────────────────────
    ax1b = ax1.twinx()
    ax1b.plot(x, df["open_pct"].values, "o--", color="#1a1aff", lw=1.5,
              markersize=5, label="Open % (line)")
    ax1b.set_ylabel("Open %", fontsize=9, color="#1a1aff")
    ax1b.tick_params(axis="y", labelcolor="#1a1aff")
    ax1b.set_ylim(0, 115)
    ax1b.legend(loc="upper left", fontsize=8)

    # ── 3. Text duplicate user % ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    bars2 = ax2.bar(x, df["dup_user_pct"].values, color="#9b59b6", alpha=0.85)
    ax2.bar_label(bars2, fmt="%.1f%%", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("% of users", fontsize=9)
    ax2.set_title("% Users with ≥1 Duplicate Review Text", fontsize=10)

    # ── 4. Photo proxy % ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    bars3 = ax3.bar(x, df["photo_pct"].values, color="#27ae60", alpha=0.85)
    ax3.bar_label(bars3, fmt="%.1f%%", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel("% of users", fontsize=9)
    ax3.set_title("% Users with Photo Compliments\n(proxy: has posted photos)", fontsize=10)

    plt.tight_layout()
    p = output_dir / f"cluster{parent}_subcluster_profile.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {p}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    sub_df = pd.read_parquet(sub_path)
    con    = duckdb.connect(str(DB_PATH), read_only=True)

    for parent in PARENT_CLUSTERS:
        logger.info(f"[Start] Cluster {parent}")
        df = analyze_parent(con, parent, sub_df)
        plot_parent(parent, df)
        logger.info(f"[Done] Cluster {parent}")

    con.close()
    logger.info("[All done]")


if __name__ == "__main__":
    main()
