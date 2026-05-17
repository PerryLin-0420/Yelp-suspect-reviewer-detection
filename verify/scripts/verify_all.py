"""
Back-Verification of Flagged Users
====================================
Takes the final removal list (3,439 users) and checks how they look
across every analysis dimension compared to normal users.

V1 — K-Means 1 cluster distribution
  Do flagged users (≥5 reviews) concentrate in anomalous K1 clusters?

V2 — Time-of-day fingerprint
  Average 24-dim hourly distribution: flagged by category vs. normal sample.

V3 — Account lifetime distribution
  Position distribution for flagged users vs. all multi-review users.

V4 — closed_pct (business hours mismatch)
  Compute for flagged + random normal sample; compare distributions.

All outputs → verify/analysis/
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger

# ── paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parents[2]
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
OUT           = Path(__file__).parents[1] / "analysis"

REMOVAL       = BASE / "lifetime kmeans" / "result" / "final" / "removal_final.parquet"
ALL_SCORES    = BASE / "K-means"         / "result" / "all_scores.parquet"
MULTI_ASSIGN  = BASE / "lifetime kmeans" / "result" / "multi_review" / "multi_assignments.parquet"
SINGLE_ASSIGN = BASE / "lifetime kmeans" / "result" / "single_review" / "single_assignments.parquet"
MULTI_STATS   = BASE / "lifetime kmeans" / "result" / "multi_review" / "cluster_stats.csv"
HOUR_MATRIX   = BASE / "user behavier matrix" / "result" / "user_hour_matrix.parquet"
K1_CLOSED     = BASE / "K-means" / "result" / "k1_closed_pct.parquet"

HOUR_COLS     = [f"h{h:02d}" for h in range(24)]
NORMAL_SAMPLE = 5000   # random non-flagged users for comparison
DAY_COLS = {
    0: ("Sunday_start_time",    "Sunday_end_time"),
    1: ("Monday_start_time",    "Monday_end_time"),
    2: ("Tuesday_start_time",   "Tuesday_end_time"),
    3: ("Wednesday_start_time", "Wednesday_end_time"),
    4: ("Thursday_start_time",  "Thursday_end_time"),
    5: ("Friday_start_time",    "Friday_end_time"),
    6: ("Saturday_start_time",  "Saturday_end_time"),
}

REASON_COLORS = {
    "single_coordinated":  "#c0392b",
    "burst_coordinated":   "#e67e22",
    "single_high_overlap": "#8e44ad",
    "k1_coordinated":      "#2980b9",
    "normal":              "#7f8c8d",
}
REASON_LABELS = {
    "single_coordinated":  "Single-review coordinated",
    "burst_coordinated":   "Burst-cluster coordinated",
    "single_high_overlap": "Single high-overlap",
    "k1_coordinated":      "K1 coordinated",
    "normal":              "Normal users (sample)",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def time_to_min(t) -> int | None:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    if hasattr(t, "hour"):
        return t.hour * 60 + t.minute
    return None


def build_hours_lookup(hours_df: pd.DataFrame) -> dict:
    lookup = {}
    for row in hours_df.itertuples(index=False):
        bid = row.business_id
        lookup[bid] = {}
        for dow, (sc, ec) in DAY_COLS.items():
            s = time_to_min(getattr(row, sc))
            e = time_to_min(getattr(row, ec))
            if s is not None and e is not None:
                if e == 0:
                    e = 1440
                lookup[bid][dow] = (s, e)
    return lookup


def compute_closed_pct_for(reviews: pd.DataFrame, hours_lookup: dict) -> pd.DataFrame:
    statuses = []
    for row in reviews.itertuples(index=False):
        bid = row.business_id
        dow = int(row.day_of_week)
        r_min = int(row.review_hour) * 60 + int(row.review_minute)
        if bid not in hours_lookup:
            statuses.append(None)
            continue
        day_hours = hours_lookup[bid].get(dow)
        if day_hours is None:
            statuses.append(True)
        else:
            s, e = day_hours
            statuses.append(not (s <= r_min < e))
    reviews = reviews.copy()
    reviews["is_closed"] = statuses
    valid = reviews[reviews["is_closed"].notna()].copy()
    valid["is_closed"] = valid["is_closed"].astype(bool)
    result = (valid.groupby("user_id")
              .agg(n_reviews=("is_closed", "count"),
                   closed=("is_closed", "sum"))
              .reset_index())
    result["closed_pct"] = result["closed"] / result["n_reviews"]
    return result


def behavior_vector(matrix: pd.DataFrame, user_ids: list) -> np.ndarray:
    eps = 1e-10
    uid_set = set(user_ids)
    sub = matrix[matrix.index.get_level_values("user_id").isin(uid_set)][HOUR_COLS].copy().astype(float)
    sub[sub == -1] = np.nan
    row_totals = sub.sum(axis=1)
    active = row_totals > 0
    norm = sub.copy()
    norm[active]  = sub[active].div(row_totals[active], axis=0)
    norm[~active] = 0.0
    norm = norm.fillna(0.0)
    acc  = norm.groupby(level="user_id").sum()
    tot  = acc.sum(axis=1)
    bvec = acc.div(tot + eps, axis=0)
    # Return mean vector across all requested users that are present
    present = [u for u in user_ids if u in bvec.index]
    if not present:
        return np.zeros(24)
    return bvec.loc[present].values.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# V1 — K-Means 1 cluster distribution
# ══════════════════════════════════════════════════════════════════════════════

def v1_k1_clusters(flagged: pd.DataFrame, all_scores: pd.DataFrame):
    logger.info("[V1] K-Means 1 cluster distribution")

    # Flagged users with K1 data (≥5 reviews, so they appear in all_scores)
    flag_k1 = flagged.merge(all_scores[["user_id", "cluster"]], on="user_id", how="inner")
    logger.info(f"  Flagged with K1 data: {len(flag_k1):,} / {len(flagged):,}")

    total_per_cluster = all_scores["cluster"].value_counts().sort_index()
    flag_per_cluster  = flag_k1["cluster"].value_counts().sort_index()
    flag_rate = (flag_per_cluster / total_per_cluster * 100).fillna(0).sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"V1 — K-Means 1 Cluster Distribution\n"
        f"Flagged users in K1 space: {len(flag_k1):,} / {len(flagged):,} "
        f"(flagged users with ≥5 reviews)",
        fontsize=11, y=1.01
    )

    ax = axes[0]
    clusters = sorted(total_per_cluster.index)
    x = np.arange(len(clusters))
    w = 0.4
    total_norm = total_per_cluster.reindex(clusters).fillna(0) / total_per_cluster.sum() * 100
    flag_norm  = flag_per_cluster.reindex(clusters).fillna(0) / len(flag_k1) * 100
    ax.bar(x - w/2, total_norm.values, w, label="All K1 users (%)", color="#4C72B0", alpha=0.7)
    ax.bar(x + w/2, flag_norm.values,  w, label="Flagged users (%)", color="#c0392b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    ax.set_xlabel("K1 Cluster", fontsize=9)
    ax.set_ylabel("% of users", fontsize=9)
    ax.set_title("Cluster share — All vs. Flagged", fontsize=9)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    colors = ["#c0392b" if r > 5 else "#4C72B0" for r in flag_rate.reindex(clusters).fillna(0)]
    ax2.bar(clusters, flag_rate.reindex(clusters).fillna(0).values, color=colors, alpha=0.85)
    ax2.axhline(len(flag_k1) / len(all_scores) * 100, color="gray", ls="--", lw=1.2,
                label=f"Overall rate {len(flag_k1)/len(all_scores)*100:.2f}%")
    ax2.set_xlabel("K1 Cluster", fontsize=9)
    ax2.set_ylabel("Flagged rate (%)", fontsize=9)
    ax2.set_title("Flagged rate per cluster (red = >5%)", fontsize=9)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "v1_k1_cluster_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("[V1] Saved v1_k1_cluster_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# V2 — Time-of-day fingerprint
# ══════════════════════════════════════════════════════════════════════════════

def v2_hourly_fingerprint(flagged: pd.DataFrame, matrix: pd.DataFrame,
                          all_user_ids: list):
    logger.info("[V2] Time-of-day fingerprint")

    all_flagged_ids = set(flagged["user_id"].tolist())
    normal_ids = [u for u in all_user_ids if u not in all_flagged_ids]
    rng = np.random.default_rng(42)
    normal_sample = rng.choice(normal_ids,
                               size=min(NORMAL_SAMPLE, len(normal_ids)),
                               replace=False).tolist()

    groups = {"normal": normal_sample}
    for reason in flagged["reason"].unique():
        groups[reason] = flagged[flagged["reason"] == reason]["user_id"].tolist()

    vectors = {name: behavior_vector(matrix, ids) for name, ids in groups.items()}
    # Exclude groups with all-zero vectors (users not in matrix)
    vectors = {k: v for k, v in vectors.items() if v.sum() > 0}

    hours = np.arange(24)
    n = len(vectors)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "V2 — Average Time-of-Day Fingerprint by Category\n"
        "(fraction of reviews per hour, averaged across users in group)",
        fontsize=11, y=1.02
    )

    for ax, (name, vec) in zip(axes, vectors.items()):
        color = REASON_COLORS.get(name, "#555")
        label = REASON_LABELS.get(name, name)
        n_users = len(groups[name])
        ax.bar(hours, vec, color=color, alpha=0.85, width=0.8)
        ax.axvspan(6, 12, alpha=0.08, color="red", label="06–11h morning zone")
        ax.set_title(f"{label}\nn={n_users:,}", fontsize=8)
        ax.set_xlabel("Hour (local)", fontsize=8)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Fraction of reviews", fontsize=8)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT / "v2_hourly_fingerprint.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("[V2] Saved v2_hourly_fingerprint.png")


# ══════════════════════════════════════════════════════════════════════════════
# V3 — Lifetime position distribution
# ══════════════════════════════════════════════════════════════════════════════

def v3_lifetime_dist(flagged: pd.DataFrame, con):
    logger.info("[V3] Lifetime position distribution")

    multi  = pd.read_parquet(MULTI_ASSIGN)
    single = pd.read_parquet(SINGLE_ASSIGN)
    stats  = pd.read_csv(MULTI_STATS)

    # Global max for normalization (same as in lifetime_kmeans.py)
    global_max = con.execute("""
        SELECT MAX(EPOCH(r.date) - EPOCH(u.yelping_since))
        FROM review r JOIN user u ON r.user_id = u.user_id
        WHERE r.date >= u.yelping_since
    """).fetchone()[0]

    # For flagged users, fetch positions
    flagged_ids = set(flagged["user_id"].tolist())
    ids_sql = ", ".join(str(i) for i in flagged_ids)
    pos_df = con.execute(f"""
        SELECT r.user_id,
               (EPOCH(r.date) - EPOCH(u.yelping_since)) / {global_max} AS position
        FROM review r JOIN user u ON r.user_id = u.user_id
        WHERE r.user_id IN ({ids_sql}) AND r.date >= u.yelping_since
    """).fetchdf()

    # Random normal sample for comparison
    all_multi_ids = set(multi["user_id"].tolist())
    normal_multi  = list(all_multi_ids - flagged_ids)
    rng = np.random.default_rng(42)
    normal_sample = rng.choice(normal_multi,
                               size=min(NORMAL_SAMPLE, len(normal_multi)),
                               replace=False).tolist()
    ns_sql = ", ".join(str(i) for i in normal_sample)
    norm_pos = con.execute(f"""
        SELECT r.user_id,
               (EPOCH(r.date) - EPOCH(u.yelping_since)) / {global_max} AS position
        FROM review r JOIN user u ON r.user_id = u.user_id
        WHERE r.user_id IN ({ns_sql}) AND r.date >= u.yelping_since
    """).fetchdf()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "V3 — Account Lifetime Position Distribution\n"
        "X = normalized position on account lifetime [0=just joined → 1=dataset end]",
        fontsize=11, y=1.01
    )
    bins = np.linspace(0, 1, 51)

    ax = axes[0]
    ax.hist(norm_pos["position"], bins=bins, density=True,
            color=REASON_COLORS["normal"], alpha=0.6, label=f"Normal sample (n={len(normal_sample):,})")
    ax.hist(pos_df["position"], bins=bins, density=True,
            color="#c0392b", alpha=0.7, label=f"Flagged (n={len(flagged_ids):,})")
    ax.set_xlabel("Lifetime position", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Flagged vs. Normal — full range", fontsize=9)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.hist(norm_pos["position"], bins=bins, density=True,
             color=REASON_COLORS["normal"], alpha=0.6, label="Normal")
    for reason, color in REASON_COLORS.items():
        if reason == "normal":
            continue
        sub_ids = set(flagged[flagged["reason"] == reason]["user_id"].tolist())
        sub_pos = pos_df[pos_df["user_id"].isin(sub_ids)]["position"]
        if len(sub_pos) > 0:
            ax2.hist(sub_pos, bins=bins, density=True, alpha=0.55,
                     color=color, label=f"{REASON_LABELS[reason]} (n={len(sub_pos):,})")
    ax2.set_xlabel("Lifetime position", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("By category", fontsize=9)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT / "v3_lifetime_dist.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("[V3] Saved v3_lifetime_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# V4 — closed_pct comparison
# ══════════════════════════════════════════════════════════════════════════════

def v4_closed_pct(flagged: pd.DataFrame, con):
    logger.info("[V4] closed_pct comparison")

    flagged_ids = set(flagged["user_id"].tolist())
    all_user_ids_db = set(
        con.execute("SELECT DISTINCT user_id FROM review").fetchdf()["user_id"].tolist()
    )
    normal_pool = list(all_user_ids_db - flagged_ids)
    rng = np.random.default_rng(42)
    normal_sample = rng.choice(normal_pool,
                               size=min(NORMAL_SAMPLE, len(normal_pool)),
                               replace=False).tolist()

    all_target = list(flagged_ids) + normal_sample
    ids_sql = ", ".join(str(i) for i in all_target)

    reviews = con.execute(f"""
        SELECT r.user_id, r.business_id,
               CAST(EXTRACT(HOUR   FROM r.date) AS INTEGER) AS review_hour,
               CAST(EXTRACT(MINUTE FROM r.date) AS INTEGER) AS review_minute,
               CAST(EXTRACT(DOW    FROM r.date) AS INTEGER) AS day_of_week
        FROM review r WHERE r.user_id IN ({ids_sql})
    """).fetchdf()

    biz_ids  = set(reviews["business_id"].unique())
    bids_sql = ", ".join(str(b) for b in biz_ids)
    hours_df = con.execute(f"""
        SELECT * FROM business_hours WHERE business_id IN ({bids_sql})
    """).fetchdf()
    hours_lookup = build_hours_lookup(hours_df)

    cp = compute_closed_pct_for(reviews, hours_lookup)
    cp["group"] = cp["user_id"].apply(
        lambda u: "flagged" if u in flagged_ids else "normal"
    )

    flagged_cp = cp[cp["group"] == "flagged"]["closed_pct"]
    normal_cp  = cp[cp["group"] == "normal"]["closed_pct"]

    # Also break flagged by reason
    reason_map = flagged.set_index("user_id")["reason"].to_dict()
    cp["reason"] = cp["user_id"].map(reason_map).fillna("normal")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "V4 — Closed-Hours Review Rate (closed_pct)\n"
        "Fraction of reviews posted when the reviewed business was closed",
        fontsize=11, y=1.01
    )
    bins = np.linspace(0, 1, 41)

    ax = axes[0]
    ax.hist(normal_cp,  bins=bins, density=True, alpha=0.65,
            color=REASON_COLORS["normal"], label=f"Normal sample (n={len(normal_cp):,})\nmean={normal_cp.mean():.2f}")
    ax.hist(flagged_cp, bins=bins, density=True, alpha=0.65,
            color="#c0392b", label=f"Flagged (n={len(flagged_cp):,})\nmean={flagged_cp.mean():.2f}")
    ax.axvline(normal_cp.mean(),  color=REASON_COLORS["normal"], ls="--", lw=1.2)
    ax.axvline(flagged_cp.mean(), color="#c0392b", ls="--", lw=1.2)
    ax.set_xlabel("closed_pct", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Flagged vs. Normal", fontsize=9)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.hist(normal_cp, bins=bins, density=True, alpha=0.5,
             color=REASON_COLORS["normal"], label=f"Normal\nmean={normal_cp.mean():.2f}")
    for reason, color in REASON_COLORS.items():
        if reason == "normal":
            continue
        sub_cp = cp[cp["reason"] == reason]["closed_pct"]
        if len(sub_cp) > 0:
            ax2.hist(sub_cp, bins=bins, density=True, alpha=0.55, color=color,
                     label=f"{REASON_LABELS[reason]}\nn={len(sub_cp):,}  mean={sub_cp.mean():.2f}")
    ax2.set_xlabel("closed_pct", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("By category", fontsize=9)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT / "v4_closed_pct.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("[V4] Saved v4_closed_pct.png")

    return normal_cp.mean(), flagged_cp.mean()


# ══════════════════════════════════════════════════════════════════════════════
# Summary overview
# ══════════════════════════════════════════════════════════════════════════════

def summary_overview(flagged: pd.DataFrame, all_scores: pd.DataFrame,
                     multi: pd.DataFrame, single: pd.DataFrame):
    logger.info("[Summary] Building overview")

    reason_counts = flagged["reason"].value_counts()

    # K1 cluster for flagged users
    flag_k1 = flagged.merge(all_scores[["user_id", "cluster", "similarity"]],
                            on="user_id", how="left")
    k1_present = flag_k1["cluster"].notna()

    # Lifetime cluster
    flag_multi  = flagged.merge(multi[["user_id", "cluster"]].rename(
        columns={"cluster": "lt_cluster"}), on="user_id", how="left")
    flag_single = flagged[flagged["user_id"].isin(set(single["user_id"]))]["user_id"].nunique()

    logger.info(
        f"\n{'='*65}\n"
        f"  VERIFICATION SUMMARY\n"
        f"{'='*65}\n"
        f"  Total flagged users                  : {len(flagged):,}\n"
        f"{'─'*65}\n"
        f"  By reason:\n"
        + "\n".join(
            f"    {r:<35}: {n:>6,}  ({n/len(flagged)*100:.1f}%)"
            for r, n in reason_counts.items()
        ) + f"\n"
        f"{'─'*65}\n"
        f"  In K1 system (≥5 reviews)            : {k1_present.sum():,} / {len(flagged):,}\n"
        f"    Avg cosine similarity              : {flag_k1.loc[k1_present,'similarity'].mean():.3f}\n"
        f"  In Lifetime single-review system     : {flag_single:,}\n"
        f"  In Lifetime multi-review system      : {flag_multi['lt_cluster'].notna().sum():,}\n"
        f"{'='*65}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("[Loading] Result files")
    flagged    = pd.read_parquet(REMOVAL)
    all_scores = pd.read_parquet(ALL_SCORES)
    multi      = pd.read_parquet(MULTI_ASSIGN)
    single     = pd.read_parquet(SINGLE_ASSIGN)
    matrix     = pd.read_parquet(HOUR_MATRIX)
    logger.info(f"  Flagged: {len(flagged):,}  |  K1 all_scores: {len(all_scores):,}  |  Matrix: {matrix.shape}")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    all_k1_ids = all_scores["user_id"].tolist()

    summary_overview(flagged, all_scores, multi, single)
    v1_k1_clusters(flagged, all_scores)
    v2_hourly_fingerprint(flagged, matrix, all_k1_ids)
    v3_lifetime_dist(flagged, con)
    norm_mean, flag_mean = v4_closed_pct(flagged, con)

    logger.info(
        f"\n[closed_pct summary]\n"
        f"  Normal sample mean : {norm_mean:.3f}\n"
        f"  Flagged mean       : {flag_mean:.3f}\n"
        f"  Δ                  : {flag_mean - norm_mean:+.3f}"
    )

    con.close()
    logger.info("\n[All done] → verify/analysis/")


if __name__ == "__main__":
    main()
