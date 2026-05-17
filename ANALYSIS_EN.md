# Yelp Suspicious User Detection

## Overview

Unsupervised anomaly detection on the Yelp Open Dataset (1,987,841 users). No ground truth labels required — anomaly signals emerge from behavioral patterns across two independent dimensions, confirmed by business-level temporal clustering.

---

## Pipeline

```
[Database]
    │
    ├─► Branch 1: Time-of-Day K-Means      Branch 2: Account Lifetime K-Means
    │   User × 24-dim hourly fingerprint       User × 50-bin lifetime density vector
    │   ↓                                      ↓
    │   Anomalous clusters (C10, C19)           Single-review / Burst accounts
    │   → Sub3/Sub4 → cosine similarity         ↓
    │   → 4,037 candidates                     Business concentration analysis
    │   ↓
    │   Business concentration (K1 Tier-2)
    │
    └─► Final removal list (3,439 users)
    │
    └─► Back-verification (verify/)
```

---

## Branch 1 — Time-of-Day K-Means

### Design

Human reviewers follow circadian rhythms. Accounts systematically posting at hours inconsistent with normal waking activity are behaviorally anomalous regardless of review content.

**Timestamp format**: Yelp stores local time (empirically confirmed — PA users peak at 18:00 Eastern local time). Morning anomaly signal is timezone-accurate.

### Features

Each user → **24-dimensional probability vector** of hourly review activity

- Per-year row normalization → summed → re-normalized (collapses year dimension)
- Users with < 5 total reviews excluded → **287,000 valid users**

### Clustering & Anomalous Cluster Identification

MiniBatchKMeans (K=20), evaluated by **entropy** and **morning ratio (06:00–11:59)**.

Clusters 10 and 19 flagged: sharply elevated morning activity, near-zero during normal waking hours.

**Cross-validation with business operating hours:**

| Sub-cluster | n | Morning reviews during open hours | Normal baseline |
|-------------|---|-----------------------------------|-----------------|
| C10-Sub3 | 420 | 9% | 43% |
| C10-Sub4 | 357 | 14% | 43% |

### Cosine Similarity Scoring

Centroid built from Sub3+Sub4. Threshold = 5th percentile of known anomaly scores (**0.546**).  
→ **4,037 anomaly candidates** (1.4% of 287k)

### K1 Tier-2 Validation

Only **9 users** appear in both K1 candidates and the Lifetime removal list — the two branches are nearly orthogonal by design.

The remaining **4,028 K1-only users** (Tier 2) were validated using the same business concentration logic:

- 45 businesses showed coordinated bursts (span < 7d) → **78 users confirmed**, added to removal list
- No business reached suspicion_score ≥ 0.5
- closed_pct mean = 0.847 (vs. 0.721 for normal users) confirms morning anomaly is real, not a timezone artifact

---

## Branch 2 — Account Lifetime K-Means

### Design

Regardless of timezone, suspicious accounts cluster review activity at a specific point in their lifetime:

- **Single-review**: one review shortly after account creation, then silent
- **Burst-then-die**: dense early activity, then gone

### Global Normalization

```
position = (review_date − yelping_since) / global_max_elapsed  (~6,241 days)
```

Maps all users to a common [0, 1] timeline regardless of join date.

### Single-Review Users (n = 1,135,945)

K-Means on 50-bin one-hot vectors.

**Key finding**: Cluster 1 (n=320,060), mean position ≈ 0.003  
→ 28% of single-review users posted their only review within ~19 days of account creation.

### Multi-Review Users (n = 851,896)

50-bin probability vectors, MiniBatchKMeans (K=20), evaluated by entropy.

| Cluster | Entropy | Peak position | Pattern |
|---------|---------|---------------|---------|
| 1 (n=91,597) | 0.31 | 0.00 | Clearest burst-then-die |

**Burst definition**: `peak_position ≤ 0.10` AND `entropy < 2.0`  
→ Clusters 1, 5, 7, 9, 14 — **217,641 users**

---

## Business Concentration & Temporal Clustering

### Logic

High suspect coverage + short time window = coordinated campaign  
High suspect coverage spread over years = natural accumulation (not a signal)

### Metrics

| Metric | Definition |
|--------|------------|
| `coverage_rate` | # suspect users / total reviews for that business |
| `span_days` | Days between first and last suspect review |
| `dense_30d_pct` | Fraction of suspect reviews in densest 30-day window |
| `suspicion_score` | `coverage_rate × dense_30d_pct` |

**Coordination threshold**: `span_days < 7`  
**High-overlap threshold**: `suspicion_score ≥ 0.5`

---

## Final Output

> Dataset: **1,987,841 users**  
> File: `lifetime kmeans/result/final/removal_final.parquet`

| Category | Count | % of Total | Criteria |
|----------|-------|-----------|----------|
| Coordinated — single-review | 1,424 | 0.072% | ≥ 2 suspect accounts, same business, span < 7d |
| Coordinated — burst-cluster | 744 | 0.037% | Same |
| Single-review, high overlap | 1,193 | 0.060% | suspicion_score ≥ 0.5 |
| K1 coordinated | 78 | 0.004% | K1 candidate + business coordination |
| **Total** | **3,439** | **0.173%** | |

---

## Back-Verification: Methodological Independence

`verify/scripts/verify_all.py` checks the 3,439 flagged users across all analysis dimensions.

| Metric | Value |
|--------|-------|
| Flagged users in K1 space (≥5 reviews) | 514 / 3,439 (15%) |
| Avg cosine similarity of those 514 | 0.168 (K1 threshold = 0.546) |
| In Lifetime single-review system | 2,617 (76%) |
| closed_pct — normal sample mean | 0.721 |
| closed_pct — flagged mean | 0.740 (Δ = +0.018) |

**Key insight**: Flagged users look normal on dimensions they were *not* selected for. This isn't a problem — it is evidence that **each detection path found a genuinely distinct class of anomaly**, not the same population through different lenses:

| Category | Behavioral Pattern | Inferred Motivation |
|----------|--------------------|---------------------|
| `single_coordinated` | Multiple accounts, same business, 7-day window, then gone | Paid one-shot campaigns |
| `burst_coordinated` | Early burst, then silent; shares business targets | Bulk operations or acquired accounts |
| `single_high_overlap` | Reviews concentrated at suspect-heavy businesses | Ecosystem participants |
| `k1_coordinated` | Systematic morning posting + shared targets | Automated scripts or remote-operated accounts |

---

## Threshold Rationale

| Threshold | Reasoning |
|-----------|-----------|
| `MIN_SUSPECT_USERS = 2` | Two independent one-shot accounts hitting the same small business within 7 days is already implausible. Requiring 3 misses real coordination. |
| `span_days < 7` | 3 days = near-certain coordination. 7 days still implausible as coincidence for new one-shot accounts. Beyond 14 days, organic word-of-mouth becomes a plausible explanation. |
| `suspicion_score ≥ 0.5` | Requires both ≥50% coverage AND ≥50% temporal concentration. Either condition alone is insufficient. |
| Burst users excluded without coordination | 217,641 burst users share the same signature as early-adopter power users. Without direct coordination evidence, removal would be speculative. Only 744 linked to a 7-day burst are included. |

---

## Detection Signal Summary

| Signal | Branch | Strength |
|--------|--------|----------|
| Morning activity 06–11h (local time) | Time-of-Day K-Means | Strong; cross-validated with business open hours |
| Business open-hours mismatch | Time-of-Day K-Means | Timezone-agnostic corroboration |
| Cosine similarity to anomaly centroid | Time-of-Day K-Means | Extends to all 287k users |
| K1 + business coordination | K-Means × Business concentration | Behavioral confirmation of K1 candidates |
| Single-review account | Lifetime K-Means | Broad; low precision alone |
| Burst-then-die lifetime | Lifetime K-Means | Moderate; strong when combined with clustering |
| Coordinated targeting (span < 7d) | Business concentration | Highest precision; smallest count |
