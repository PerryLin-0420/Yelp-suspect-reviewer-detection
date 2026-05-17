# Yelp Low-Credibility Reviewer Detection

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![DuckDB](https://img.shields.io/badge/DuckDB-0.10+-yellow)

Unsupervised detection of low-credibility reviewers on the [Yelp Open Dataset](https://www.yelp.com/dataset) (1,987,841 users). No labeled data required.

Finds behaviorally anomalous accounts across two independent dimensions, then confirms coordinated activity through business-level temporal clustering. The output is a **behavioral filter** — not a claim to have caught every fake account.

> Side project. Not formal research.  
> 繁體中文說明：[README_MANDARIN.md](README_MANDARIN.md)

---

## Prerequisites — Database

All scripts read from a single `YELP.duckdb`. Tables needed: `review`, `user`, `business`, `business_hours`.

There is no pre-built database to download — you need to build it yourself:

1. Download the raw JSON from [Yelp Open Dataset](https://www.yelp.com/dataset) (academic license, free)
2. Import the four entities into DuckDB — the schema must match the column names used in these scripts. The database build scripts are in a separate repo; each entity has its own `autobuild_db.bat` that populates the relevant table once the JSON is in place. Manual import works too.
3. Copy `config.example.py` to `config.py` and point it to your database file:

```bash
cp config.example.py config.py
# then edit config.py: set DB_PATH to your YELP.duckdb location
```

All scripts import `DB_PATH` from `config.py` — it's the only path you need to change. `config.py` is gitignored and never committed.

---

## How It Works

### Branch 1 — Time-of-Day K-Means

Each user is represented as a **24-dim probability vector** of hourly review activity. Normal people have routines; posting heavily at unusual hours is a signal in itself.

- Yelp stores local timestamps (verified empirically — PA users peak at 18:00 Eastern)
- MiniBatchKMeans (K=20) identifies clusters with abnormally high morning activity (06:00–11:59)
- Sub-clustering isolates the most extreme groups: Sub3/Sub4 have only **9–14%** of morning reviews falling during business open hours (vs. 43% for normal clusters)
- Cosine similarity against the Sub3/Sub4 centroid → **4,037 candidates** across all 287k users

### Branch 2 — Account Lifetime K-Means

Each user's reviews mapped onto a normalized account lifetime **[0, 1]**, represented as a **50-bin density vector**.

- Single-review users (n=1,135,945): 28% posted their only review within ~19 days of account creation, then vanished
- Multi-review users (n=851,896): burst-then-die pattern identified by low entropy and peak at position 0.00

### Business Concentration Confirmation

Candidate accounts alone aren't enough — they also need to cluster on the same businesses.

| Metric | Definition |
|--------|------------|
| `coverage_rate` | # suspect accounts / total reviews for that business |
| `span_days` | Days between first and last suspect review |
| `dense_30d_pct` | Fraction of suspect reviews in the densest 30-day window |
| `suspicion_score` | `coverage_rate × dense_30d_pct` |

Coordinated burst: `span_days < 7` — High overlap: `suspicion_score ≥ 0.5`

Full methodology: [ANALYSIS_EN.md](ANALYSIS_EN.md) | [ANALYSIS_MANDARIN.md](ANALYSIS_MANDARIN.md)  
Execution order: [PIPELINE.md](PIPELINE.md)

---

## Results

| Category | Count | % of Total |
|----------|-------|-----------|
| Coordinated — single-review accounts | 1,424 | 0.072% |
| Coordinated — burst-cluster accounts | 744 | 0.037% |
| Single-review, high business overlap | 1,193 | 0.060% |
| Time-of-day anomaly + coordinated | 78 | 0.004% |
| **Total flagged** | **3,439** | **0.173%** |

Output: `lifetime kmeans/result/final/removal_final.parquet` — columns: `user_id`, `reason`

---

## Characteristics & Limitations

**What it does:**
- Behavioral anomaly detection without labeled data or content review
- Two independent signals — each targets a different type of suspicious behavior
- Conservative: only accounts with direct coordination evidence are flagged

**What it doesn't guarantee:**
- No precision / recall — unsupervised, no ground truth
- Accounts with normal behavioral patterns won't be caught (e.g., slow drip campaigns)
- The same behavioral pattern can have legitimate explanations

### Back-Verification: Two Branches, Two Different Populations

Projecting the 3,439 flagged users back onto all dimensions shows they look normal on the dimensions they weren't selected for (avg K1 cosine similarity 0.168 vs threshold 0.546; closed_pct Δ = +0.018).

That's not a problem — it's evidence that **each detection path found a genuinely distinct behavioral class**:

| Category | Pattern | Inferred Motivation |
|----------|---------|---------------------|
| `single_coordinated` | Multiple accounts hit the same business within 7 days, then disappear | Paid one-shot campaigns |
| `burst_coordinated` | Dense early activity, shared business targets, then silent | Bulk operations or acquired accounts |
| `single_high_overlap` | Reviews concentrated at suspect-heavy businesses | Ecosystem participants |
| `k1_coordinated` | Systematic morning posting + shared targets | Automated scripts or remote-operated accounts |

---

## Structure

```
├── README.md
├── README_MANDARIN.md
├── PIPELINE.md
├── ANALYSIS_EN.md / ANALYSIS_MANDARIN.md
│
├── user behavier matrix/scripts/   # Step 0: build 24-dim hourly matrix
├── K-means/scripts/                # Steps 1–6, 9–10: time-of-day analysis
├── lifetime kmeans/scripts/        # Steps 7–8: lifetime analysis
└── verify/scripts/                 # Step 11: back-verification
```

---

## Requirements

```bash
pip install duckdb pandas numpy scikit-learn matplotlib scipy loguru
```
