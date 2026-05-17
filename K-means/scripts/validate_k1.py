"""
K-Means 1 Tier-2 Validation via Business Concentration
=======================================================
K1 candidates (time-of-day anomaly, cosine similarity > threshold) minus
Lifetime removal list = 4,028 users with no burst/coordination evidence yet.

This script applies the same business concentration analysis used in
business_concentration.py to validate whether K1 users also show
coordinated review behavior on specific businesses.

Outputs:
  K-means/result/k1_validated.parquet  — K1 users confirmed by coordination
  K-means/analysis/k1_biz_coverage.png
  K-means/analysis/k1_biz_coverage_timeline.png
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from loguru import logger

DB_PATH       = Path("E:/Project/Yelp/database/YELP.duckdb")
K1_DIR        = Path(__file__).parents[1]
LT_DIR        = K1_DIR.parent / "lifetime kmeans"
ANALYSIS_OUT  = K1_DIR / "analysis"
RESULT_OUT    = K1_DIR / "result"
FINAL_DIR     = LT_DIR / "result" / "final"

MIN_TOTAL_REVIEWS  = 5
MIN_SUSPECT_USERS  = 2
CONFIRMED_SPAN_MAX = 7.0
HIGH_OVERLAP_SCORE = 0.5
DENSE_WINDOW_DAYS  = 30
TOP_SHOW           = 30
TOP_TIMELINE       = 10


# ── helpers (same as business_concentration.py) ───────────────────────────────

def dense_window_pct(dates: np.ndarray, window_days: int) -> float:
    if len(dates) == 0:
        return 0.0
    d = np.sort(dates.astype("datetime64[D]").astype("int64"))
    w = window_days
    best, j = 0, 0
    for i in range(len(d)):
        while j < len(d) and d[j] - d[i] <= w:
            j += 1
        best = max(best, j - i)
    return best / len(d)


def compute_temporal_stats(suspect_reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bid, grp in suspect_reviews.groupby("business_id"):
        dates = grp["date"].values
        if len(dates) < 2:
            span, d30 = 0.0, 1.0
        else:
            span = float((dates.max() - dates.min()) / np.timedelta64(1, "D"))
            d30  = dense_window_pct(dates, DENSE_WINDOW_DAYS)
        rows.append({"business_id": bid, "span_days": span, "dense_30d_pct": d30})
    return pd.DataFrame(rows)


def build_full_df(con, suspect_ids: set,
                  total_df: pd.DataFrame, biz_df: pd.DataFrame) -> pd.DataFrame:
    ids_sql = ", ".join(str(i) for i in suspect_ids)

    cov = con.execute(f"""
        SELECT business_id, COUNT(DISTINCT user_id) AS n_suspect_users
        FROM review WHERE user_id IN ({ids_sql})
        GROUP BY business_id
    """).fetchdf()

    rev_dates = con.execute(f"""
        SELECT business_id, date FROM review WHERE user_id IN ({ids_sql})
    """).fetchdf()
    rev_dates["date"] = pd.to_datetime(rev_dates["date"])

    temporal = compute_temporal_stats(rev_dates)

    df = total_df.merge(cov,      on="business_id", how="inner")
    df = df.merge(temporal,       on="business_id", how="left")
    df = df.merge(biz_df,         on="business_id", how="left")

    df["coverage_rate"]   = df["n_suspect_users"] / df["total_reviews"]
    df["suspicion_score"] = df["coverage_rate"] * df["dense_30d_pct"]

    df = df[
        (df["total_reviews"]   >= MIN_TOTAL_REVIEWS) &
        (df["n_suspect_users"] >= MIN_SUSPECT_USERS)
    ].sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    return df


def get_confirmed_users(con, suspect_ids: set, coverage_df: pd.DataFrame) -> set:
    confirmed_bids = set(
        coverage_df[coverage_df["span_days"] < CONFIRMED_SPAN_MAX]["business_id"].tolist()
    )
    if not confirmed_bids:
        return set()
    ids_sql  = ", ".join(str(i) for i in suspect_ids)
    bids_sql = ", ".join(str(b) for b in confirmed_bids)
    df = con.execute(f"""
        SELECT DISTINCT user_id FROM review
        WHERE user_id IN ({ids_sql}) AND business_id IN ({bids_sql})
    """).fetchdf()
    return set(df["user_id"].tolist())


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_main(df: pd.DataFrame, out_path: Path):
    top = df.head(TOP_SHOW)
    labels = [
        f"{str(r.biz_name)[:30]}…" if len(str(r.biz_name)) > 30 else str(r.biz_name)
        for r in top.itertuples()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle(
        "K-Means 1 Tier-2 Users — Business Coverage + Temporal Clustering\n"
        f"suspicion_score = coverage × dense_30d  |  red = span < {CONFIRMED_SPAN_MAX:.0f}d (confirmed coordinated)",
        fontsize=11, y=0.98
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.97, wspace=0.35)

    ax = axes[0]
    colors = ["#c0392b" if r.span_days < CONFIRMED_SPAN_MAX else "#DD8452"
              for r in top.itertuples()]
    ax.barh(range(len(top)), top["suspicion_score"].values, color=colors, alpha=0.88)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    for i, r in enumerate(top.itertuples()):
        ax.text(r.suspicion_score + 0.01, i,
                f"cov={r.coverage_rate*100:.0f}%  d30={r.dense_30d_pct*100:.0f}%  span={r.span_days:.1f}d",
                va="center", fontsize=6.5)
    ax.set_xlabel("Suspicion score  (coverage × dense_30d)", fontsize=9)
    ax.set_title(
        f"Top {TOP_SHOW} by Suspicion Score\n"
        f"(red = span < {CONFIRMED_SPAN_MAX:.0f}d → confirmed coordinated)",
        fontsize=9, pad=12
    )
    ax.set_xlim(0, 1.35)

    ax2 = axes[1]
    sc = ax2.scatter(
        df["span_days"], df["coverage_rate"] * 100,
        c=df["dense_30d_pct"], cmap="RdYlGn_r",
        s=np.clip(df["n_suspect_users"] * 2, 5, 80),
        alpha=0.35, vmin=0, vmax=1
    )
    ax2.scatter(
        top["span_days"], top["coverage_rate"] * 100,
        c=top["dense_30d_pct"], cmap="RdYlGn_r",
        s=60, alpha=0.95, edgecolors="black", linewidths=0.5,
        vmin=0, vmax=1, zorder=5
    )
    cb = plt.colorbar(sc, ax=ax2)
    cb.set_label("dense_30d_pct", fontsize=8)
    ax2.set_xlabel("Span days", fontsize=9)
    ax2.set_ylabel("Coverage rate (%)", fontsize=9)
    ax2.set_title(
        "Coverage vs Temporal Spread\n(red=clustered, green=spread | size=n_suspect)",
        fontsize=9, pad=12
    )
    ax2.axvline(CONFIRMED_SPAN_MAX, color="red", ls="--", lw=1.2, alpha=0.7,
                label=f"Confirmed ({CONFIRMED_SPAN_MAX:.0f}d)")
    ax2.axvline(30, color="orange", ls=":", lw=1, alpha=0.6, label="30d line")
    ax2.legend(fontsize=8)

    ax3 = axes[2]
    ax3.hist(df["dense_30d_pct"], bins=50, color="#4C72B0", alpha=0.85)
    ax3.axvline(df["dense_30d_pct"].mean(), color="red", ls="--", lw=1.2,
                label=f"Mean = {df['dense_30d_pct'].mean():.2f}")
    ax3.set_xlabel("dense_30d_pct", fontsize=9)
    ax3.set_ylabel("# businesses", fontsize=9)
    ax3.set_title(
        "Temporal Concentration Distribution\n(fraction in densest 30-day window)",
        fontsize=9, pad=12
    )
    ax3.legend(fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {out_path.name}")


def plot_timelines(con, top_bids: list, suspect_ids: set,
                   coverage_df: pd.DataFrame, out_path: Path):
    bids_sql = ", ".join(str(b) for b in top_bids)
    tl = con.execute(f"""
        SELECT r.business_id, r.user_id, r.date, b.name AS biz_name
        FROM review r
        JOIN business b ON r.business_id = b.business_id
        WHERE r.business_id IN ({bids_sql})
        ORDER BY r.date
    """).fetchdf()
    tl["date"]       = pd.to_datetime(tl["date"])
    tl["is_suspect"] = tl["user_id"].isin(suspect_ids)

    n_cols = 2
    n_rows = (len(top_bids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.8))
    fig.suptitle(
        f"Top {len(top_bids)} Suspicious Businesses — Review Timeline (K1 Tier-2)\n"
        "Orange = K1 morning-anomaly users  |  Blue = all others",
        fontsize=11, y=0.99
    )
    fig.subplots_adjust(top=0.93, hspace=0.5, wspace=0.3)

    for idx, bid in enumerate(top_bids):
        ax  = axes[idx // n_cols][idx % n_cols]
        sub = tl[tl["business_id"] == bid]
        if sub.empty:
            ax.set_visible(False)
            continue

        biz_name = sub["biz_name"].iloc[0]
        all_m = sub.set_index("date").resample("ME").size()
        sus_m = (sub[sub["is_suspect"]]
                 .set_index("date").resample("ME").size()
                 .reindex(all_m.index, fill_value=0))

        ax.bar(all_m.index, all_m.values, width=25, color="#4C72B0", alpha=0.5, label="All")
        ax.bar(sus_m.index, sus_m.values, width=25, color="#DD8452", alpha=0.88, label="K1 Tier-2")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

        row = coverage_df[coverage_df["business_id"] == bid]
        if len(row):
            r = row.iloc[0]
            confirmed = "★CONFIRMED" if r["span_days"] < CONFIRMED_SPAN_MAX else ""
            ax.set_title(
                f"{confirmed} {str(biz_name)[:38]}\n"
                f"score={r['suspicion_score']:.2f}  cov={r['coverage_rate']*100:.0f}%  "
                f"d30={r['dense_30d_pct']*100:.0f}%  span={r['span_days']:.1f}d",
                fontsize=8
            )
        ax.set_ylabel("Reviews/month", fontsize=7)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(top_bids), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {out_path.name}")


# ── merge into final removal list ─────────────────────────────────────────────

def merge_into_final(k1_coordinated: set, k1_high_overlap: set):
    existing = pd.read_parquet(FINAL_DIR / "removal_final.parquet")
    existing_ids = set(existing["user_id"].tolist())

    new_coord   = k1_coordinated  - existing_ids
    new_overlap = k1_high_overlap - existing_ids - k1_coordinated

    additions = []
    for uid in sorted(new_coord):
        additions.append({"user_id": uid, "reason": "k1_coordinated"})
    for uid in sorted(new_overlap):
        additions.append({"user_id": uid, "reason": "k1_high_overlap"})

    if not additions:
        logger.info("[Merge] No new users to add — K1 validated set fully overlaps with existing")
        return existing

    additions_df = pd.DataFrame(additions)
    merged = pd.concat([existing, additions_df], ignore_index=True)
    merged.to_parquet(FINAL_DIR / "removal_final.parquet", index=False)

    logger.info(
        f"\n{'='*60}\n"
        f"  Existing removal list                   : {len(existing_ids):>8,}\n"
        f"  K1 coordinated (new)                    : {len(new_coord):>8,}\n"
        f"  K1 high overlap (new)                   : {len(new_overlap):>8,}\n"
        f"  ─────────────────────────────────────────────────────\n"
        f"  Updated removal list                    : {len(merged):>8,}\n"
        f"{'='*60}"
    )
    return merged


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    RESULT_OUT.mkdir(parents=True, exist_ok=True)

    # Load K1 Tier-2: K1 candidates not already in Lifetime removal list
    k1_all    = pd.read_parquet(RESULT_OUT / "anomaly_candidates.parquet")
    lifetime  = pd.read_parquet(FINAL_DIR  / "removal_final.parquet")
    tier2_ids = set(k1_all["user_id"].tolist()) - set(lifetime["user_id"].tolist())
    logger.info(f"[K1 Tier-2] {len(tier2_ids):,} users to validate")

    con      = duckdb.connect(str(DB_PATH), read_only=True)
    total_df = con.execute("""
        SELECT business_id, COUNT(*) AS total_reviews FROM review GROUP BY business_id
    """).fetchdf()
    biz_df = con.execute("""
        SELECT business_id, name AS biz_name, city, state FROM business
    """).fetchdf()

    df = build_full_df(con, tier2_ids, total_df, biz_df)
    logger.info(
        f"[K1 Tier-2] {len(df):,} businesses with ≥{MIN_SUSPECT_USERS} suspect users\n"
        f"  span < {CONFIRMED_SPAN_MAX:.0f}d (coordinated): {(df['span_days'] < CONFIRMED_SPAN_MAX).sum()} businesses\n"
        f"  suspicion_score ≥ {HIGH_OVERLAP_SCORE} (high overlap): "
        f"{(df['suspicion_score'] >= HIGH_OVERLAP_SCORE).sum()} businesses"
    )
    logger.info(
        f"\n[Top 20 by suspicion_score]\n"
        f"{df.head(20)[['biz_name','city','total_reviews','n_suspect_users','coverage_rate','span_days','dense_30d_pct','suspicion_score']].to_string(index=False)}"
    )

    df.to_csv(RESULT_OUT / "k1_biz_coverage.csv", index=False)
    plot_main(df, ANALYSIS_OUT / "k1_biz_coverage.png")

    top_bids = df.head(TOP_TIMELINE)["business_id"].tolist()
    plot_timelines(con, top_bids, tier2_ids, df,
                   ANALYSIS_OUT / "k1_biz_coverage_timeline.png")

    k1_coordinated = get_confirmed_users(con, tier2_ids, df)
    logger.info(f"[K1 Tier-2] Coordinated (span<{CONFIRMED_SPAN_MAX}d): {len(k1_coordinated):,} users")

    # High-overlap: users in businesses with suspicion_score >= threshold
    high_bids = set(df[df["suspicion_score"] >= HIGH_OVERLAP_SCORE]["business_id"].tolist())
    if high_bids:
        ids_sql  = ", ".join(str(i) for i in tier2_ids)
        bids_sql = ", ".join(str(b) for b in high_bids)
        k1_high_overlap = set(con.execute(f"""
            SELECT DISTINCT user_id FROM review
            WHERE user_id IN ({ids_sql}) AND business_id IN ({bids_sql})
        """).fetchdf()["user_id"].tolist())
    else:
        k1_high_overlap = set()
    logger.info(f"[K1 Tier-2] High overlap (score≥{HIGH_OVERLAP_SCORE}): {len(k1_high_overlap):,} users")

    # Save K1 validated users separately
    k1_validated_ids = k1_coordinated | k1_high_overlap
    pd.DataFrame({
        "user_id": sorted(k1_validated_ids),
        "reason":  [
            "k1_coordinated" if u in k1_coordinated else "k1_high_overlap"
            for u in sorted(k1_validated_ids)
        ]
    }).to_parquet(RESULT_OUT / "k1_validated.parquet", index=False)
    logger.info(f"[K1 Tier-2] Total validated: {len(k1_validated_ids):,} users")

    merge_into_final(k1_coordinated, k1_high_overlap)
    con.close()
    logger.info("[All done]")


if __name__ == "__main__":
    main()
