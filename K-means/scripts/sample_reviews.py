import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
candidates_path = Path(__file__).parents[1] / "result" / "anomaly_candidates.parquet"
output_path    = Path(__file__).parents[1] / "result" / "sample_reviews.csv"

N_SAMPLE = 30

candidates = pd.read_parquet(candidates_path)
sampled_users = candidates.sample(n=N_SAMPLE, random_state=42)["user_id"].tolist()

con = duckdb.connect(str(DB_PATH), read_only=True)
ids_sql = ", ".join(str(i) for i in sampled_users)

df = con.execute(f"""
    SELECT
        u.user_id,
        u.user_raw_id,
        u.name,
        u.review_count,
        u.yelping_since,
        r.stars,
        r.date,
        EXTRACT(HOUR FROM r.date) AS review_hour,
        b.name AS business_name,
        r.text
    FROM review r
    JOIN user u ON r.user_id = u.user_id
    JOIN business b ON r.business_id = b.business_id
    WHERE r.user_id IN ({ids_sql})
    ORDER BY u.user_id, r.date
""").fetchdf()
con.close()

# One representative review per user (closest to their peak suspicious hour: 06-11)
def pick_rep(grp):
    morning = grp[grp["review_hour"].between(6, 11)]
    return morning.iloc[0] if len(morning) > 0 else grp.iloc[0]

sampled = df.groupby("user_id", group_keys=False).apply(pick_rep).reset_index(drop=True)
sampled = sampled.merge(
    candidates[["user_id", "similarity"]],
    on="user_id", how="left"
).sort_values("similarity", ascending=False)

sampled.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Saved {len(sampled)} rows → {output_path}")
