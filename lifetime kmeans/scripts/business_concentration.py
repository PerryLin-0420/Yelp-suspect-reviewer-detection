"""
Business Coverage + Temporal Clustering Analysis
==================================================
Real signal: suspect reviews are BOTH high-coverage AND temporally clustered.

Metrics per business (for suspect users):
  coverage_rate   = n_suspect_users / total_reviews
  span_days       = (last - first suspect review) in days
  dense_30d_pct   = fraction of suspect reviews in the densest 30-day window
  suspicion_score = coverage_rate * dense_30d_pct

Final ranking is by suspicion_score.
Threshold: span_days < 3  →  confirmed coordinated fake reviews.

Final outputs (result/final/):
  removal_confirmed.parquet  — users linked to businesses with span_days < 3
  removal_all.parquet        — all single-review + all burst-cluster users (broad)
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from loguru import logger

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
BASE_DIR      = Path(__file__).parents[1]
SINGLE_RESULT = BASE_DIR / "result"   / "single_review"
SINGLE_OUT    = BASE_DIR / "analysis" / "single_review"
MULTI_RESULT  = BASE_DIR / "result"   / "multi_review"
MULTI_OUT     = BASE_DIR / "analysis" / "multi_review"
FINAL_RESULT  = BASE_DIR / "result"   / "final"

MIN_TOTAL_REVIEWS  = 5
MIN_SUSPECT_USERS  = 2
TOP_SHOW           = 30
TOP_TIMELINE       = 10
DENSE_WINDOW_DAYS  = 30
CONFIRMED_SPAN_MAX = 7.0   # days — confirmed coordinated burst

BURST_PEAK_MAX    = 0.10
BURST_ENTROPY_MAX = 2.0


# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_total_reviews(con) -> pd.DataFrame:
    return con.execute("""
        SELECT business_id, COUNT(*) AS total_reviews
        FROM review GROUP BY business_id
    """).fetchdf()


def fetch_biz_names(con) -> pd.DataFrame:
    return con.execute("""
        SELECT business_id, name AS biz_name, city, state FROM business
    """).fetchdf()


def dense_window_pct(dates: np.ndarray, window_days: int) -> float:
    """Fraction of reviews in the densest `window_days`-day sliding window."""
    if len(dates) == 0:
        return 0.0
    # Convert to integer days (avoids datetime64 unit ambiguity us vs ns)
    d = np.sort(dates.astype("datetime64[D]").astype("int64"))
    w = window_days
    best = 0
    j = 0
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
            span = 0.0
            d30  = 1.0
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
    """Users who reviewed a business with span_days < CONFIRMED_SPAN_MAX."""
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

def plot_main(df: pd.DataFrame, title: str, out_path: Path):
    top = df.head(TOP_SHOW)
    labels = [
        f"{str(r.biz_name)[:30]}…" if len(str(r.biz_name)) > 30 else str(r.biz_name)
        for r in top.itertuples()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.97, wspace=0.35)

    # Left: suspicion score bar
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
        f"(red = span < {CONFIRMED_SPAN_MAX:.0f} days → confirmed coordinated)",
        fontsize=9, pad=12
    )
    ax.set_xlim(0, 1.35)

    # Middle: coverage vs span scatter
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
    ax2.set_xlabel("Span days (suspect reviews spread)", fontsize=9)
    ax2.set_ylabel("Coverage rate (%)", fontsize=9)
    ax2.set_title(
        "Coverage vs Temporal Spread\n(red=clustered, green=spread | size=overlap count)",
        fontsize=9, pad=12
    )
    ax2.axvline(CONFIRMED_SPAN_MAX, color="red", ls="--", lw=1.2, alpha=0.7,
                label=f"Confirmed threshold ({CONFIRMED_SPAN_MAX:.0f}d)")
    ax2.axvline(30, color="orange", ls=":", lw=1, alpha=0.6, label="30-day line")
    ax2.legend(fontsize=8)

    # Right: dense_30d_pct distribution
    ax3 = axes[2]
    ax3.hist(df["dense_30d_pct"], bins=50, color="#4C72B0", alpha=0.85)
    ax3.axvline(df["dense_30d_pct"].mean(), color="red", ls="--", lw=1.2,
                label=f"Mean = {df['dense_30d_pct'].mean():.2f}")
    ax3.set_xlabel("dense_30d_pct", fontsize=9)
    ax3.set_ylabel("# businesses", fontsize=9)
    ax3.set_title(
        "Temporal Concentration Distribution\n(fraction of suspect reviews in densest 30-day window)",
        fontsize=9, pad=12
    )
    ax3.legend(fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[Saved] {out_path.name}")


def plot_timelines(con, top_bids: list, suspect_ids: set,
                   coverage_df: pd.DataFrame, out_path: Path,
                   suspect_label: str):
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
        f"Top {len(top_bids)} Suspicious Businesses — Review Timeline\n"
        f"Orange = {suspect_label}  |  Blue = all other users  "
        f"(ranked by suspicion_score = coverage × dense_30d)",
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
        ax.bar(sus_m.index, sus_m.values, width=25, color="#DD8452", alpha=0.88, label=suspect_label)
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


# ── single-review ─────────────────────────────────────────────────────────────

def analyze_single(con, total_df, biz_df) -> tuple[set, set, pd.DataFrame]:
    """Returns (all_single_ids, confirmed_ids, coverage_df)."""
    logger.info("[Single] Loading assignments")
    assigns    = pd.read_parquet(SINGLE_RESULT / "single_assignments.parquet")
    single_ids = set(assigns["user_id"].tolist())
    logger.info(f"[Single] {len(single_ids):,} users")

    df = build_full_df(con, single_ids, total_df, biz_df)
    logger.info(f"[Single] {len(df):,} businesses | "
                f"span<3d: {(df['span_days'] < CONFIRMED_SPAN_MAX).sum()} businesses")
    logger.info(
        f"\n[Top 20 by suspicion_score]\n"
        f"{df.head(20)[['biz_name','city','total_reviews','n_suspect_users','coverage_rate','span_days','dense_30d_pct','suspicion_score']].to_string(index=False)}"
    )

    df.to_csv(SINGLE_RESULT / "biz_coverage.csv", index=False)

    plot_main(
        df,
        "Single-Review Users — Business Coverage + Temporal Clustering\n"
        f"suspicion_score = coverage × dense_30d  |  red bars = span < {CONFIRMED_SPAN_MAX:.0f} days (confirmed coordinated)",
        SINGLE_OUT / "biz_coverage.png"
    )

    top_bids = df.head(TOP_TIMELINE)["business_id"].tolist()
    plot_timelines(con, top_bids, single_ids, df,
                   SINGLE_OUT / "biz_coverage_timeline.png",
                   "Single-review users")

    confirmed = get_confirmed_users(con, single_ids, df)
    logger.info(f"[Single] Confirmed (span<{CONFIRMED_SPAN_MAX}d): {len(confirmed):,} users")
    logger.info("[Single] Done")
    return single_ids, confirmed, df


# ── multi burst ───────────────────────────────────────────────────────────────

def analyze_multi_burst(con, total_df, biz_df) -> tuple[set, set]:
    """Returns (all_burst_ids, confirmed_ids)."""
    logger.info("[Multi-Burst] Loading assignments + stats")
    assigns = pd.read_parquet(MULTI_RESULT / "multi_assignments.parquet")
    stats   = pd.read_csv(MULTI_RESULT / "cluster_stats.csv")

    burst_clusters = stats[
        (stats["peak_position"] <= BURST_PEAK_MAX) &
        (stats["entropy"]       <  BURST_ENTROPY_MAX)
    ]["cluster"].tolist()
    logger.info(f"[Multi-Burst] Burst clusters: {burst_clusters}")

    burst_ids = set(assigns[assigns["cluster"].isin(burst_clusters)]["user_id"].tolist())
    logger.info(f"[Multi-Burst] {len(burst_ids):,} users")

    df = build_full_df(con, burst_ids, total_df, biz_df)
    logger.info(f"[Multi-Burst] {len(df):,} businesses | "
                f"span<3d: {(df['span_days'] < CONFIRMED_SPAN_MAX).sum()} businesses")
    logger.info(
        f"\n[Top 20 by suspicion_score]\n"
        f"{df.head(20)[['biz_name','city','total_reviews','n_suspect_users','coverage_rate','span_days','dense_30d_pct','suspicion_score']].to_string(index=False)}"
    )

    df.to_csv(MULTI_RESULT / "burst_biz_coverage.csv", index=False)

    plot_main(
        df,
        f"Multi-Review Burst Users (Clusters {burst_clusters}) — Coverage + Temporal Clustering\n"
        f"suspicion_score = coverage × dense_30d  |  red bars = span < {CONFIRMED_SPAN_MAX:.0f} days (confirmed coordinated)",
        MULTI_OUT / "burst_biz_coverage.png"
    )

    top_bids = df.head(TOP_TIMELINE)["business_id"].tolist()
    plot_timelines(con, top_bids, burst_ids, df,
                   MULTI_OUT / "burst_biz_coverage_timeline.png",
                   "Burst-cluster users")

    confirmed = get_confirmed_users(con, burst_ids, df)
    logger.info(f"[Multi-Burst] Confirmed (span<{CONFIRMED_SPAN_MAX}d): {len(confirmed):,} users")
    logger.info("[Multi-Burst] Done")
    return burst_ids, confirmed


HIGH_OVERLAP_SCORE = 0.5   # suspicion_score threshold for single-review high-overlap


# ── final result export ───────────────────────────────────────────────────────

def export_final(con,
                 single_ids: set, single_confirmed: set,
                 burst_confirmed: set,
                 single_coverage_df: pd.DataFrame):
    FINAL_RESULT.mkdir(parents=True, exist_ok=True)

    # ── Group 1: coordinated fake reviews (span < 3d, single + burst) ─────────
    coordinated = single_confirmed | burst_confirmed
    coord_df = pd.DataFrame({
        "user_id": sorted(coordinated),
        "reason":  [
            "single_coordinated" if u in single_confirmed and u not in burst_confirmed
            else "burst_coordinated" if u not in single_confirmed
            else "single+burst_coordinated"
            for u in sorted(coordinated)
        ]
    })

    # ── Group 2: single-review with high business overlap ─────────────────────
    # Businesses where suspicion_score >= HIGH_OVERLAP_SCORE (coverage * dense_30d)
    high_overlap_bids = set(
        single_coverage_df[
            single_coverage_df["suspicion_score"] >= HIGH_OVERLAP_SCORE
        ]["business_id"].tolist()
    )
    ids_sql  = ", ".join(str(i) for i in single_ids)
    bids_sql = ", ".join(str(b) for b in high_overlap_bids)
    high_overlap_users = set(
        con.execute(f"""
            SELECT DISTINCT user_id FROM review
            WHERE user_id IN ({ids_sql}) AND business_id IN ({bids_sql})
        """).fetchdf()["user_id"].tolist()
    )
    # Exclude already captured by coordinated
    high_overlap_only = high_overlap_users - coordinated
    overlap_df = pd.DataFrame({
        "user_id": sorted(high_overlap_only),
        "reason":  ["single_high_overlap"] * len(high_overlap_only)
    })

    # ── Combine & save ────────────────────────────────────────────────────────
    final_df = pd.concat([coord_df, overlap_df], ignore_index=True)
    final_df.to_parquet(FINAL_RESULT / "removal_final.parquet", index=False)

    logger.info(
        f"\n{'='*60}\n"
        f"  Coordinated (span < {CONFIRMED_SPAN_MAX:.0f}d)              : {len(coordinated):>8,}\n"
        f"    – single-review                       : {len(single_confirmed):>8,}\n"
        f"    – burst-cluster                       : {len(burst_confirmed):>8,}\n"
        f"  Single-review high overlap (score≥{HIGH_OVERLAP_SCORE}) : {len(high_overlap_only):>8,}\n"
        f"  ─────────────────────────────────────────────────────\n"
        f"  Final removal list                      : {len(final_df):>8,}\n"
        f"{'='*60}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    con      = duckdb.connect(str(DB_PATH), read_only=True)
    total_df = fetch_total_reviews(con)
    biz_df   = fetch_biz_names(con)

    single_ids, single_confirmed, single_cov_df = analyze_single(con, total_df, biz_df)
    _,          burst_confirmed                  = analyze_multi_burst(con, total_df, biz_df)

    export_final(con, single_ids, single_confirmed, burst_confirmed, single_cov_df)

    con.close()
    logger.info("[All done]")


if __name__ == "__main__":
    main()
