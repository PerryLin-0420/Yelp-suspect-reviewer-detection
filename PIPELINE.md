# 執行順序與腳本說明

## Pipeline 執行順序

```
Step 0  user behavier matrix/scripts/matrix_calculator.py
        → user behavier matrix/result/user_hour_matrix.parquet

Step 1  K-means/scripts/behavior_kmeans.py
        → K-means/analysis/kmeans_clustering/kmeans_assignments.parquet

Step 2  K-means/scripts/subclustering.py
        → K-means/analysis/subclustering/subclustering_assignments.parquet

Step 3  K-means/scripts/suspicious_cluster_analysis.py
        → 視覺化（無新 parquet）

Step 4  K-means/scripts/business_hours_heatmap.py
        → K-means/analysis/deep_profile/（7×24 熱力圖）

Step 5  K-means/scripts/deep_profile_analysis.py
        → 視覺化

Step 6  K-means/scripts/similarity_scoring.py
        → K-means/result/all_scores.parquet
        → K-means/result/anomaly_candidates.parquet   ← K1 最終候選人

Step 7  lifetime kmeans/scripts/lifetime_kmeans.py
        → lifetime kmeans/result/single_review/single_assignments.parquet
        → lifetime kmeans/result/multi_review/multi_assignments.parquet

Step 8  lifetime kmeans/scripts/business_concentration.py
        → lifetime kmeans/result/final/removal_final.parquet   ← 主要除名清單

Step 9  K-means/scripts/validate_k1.py
        → K-means/result/k1_validated.parquet
        → 更新 removal_final.parquet（+78 人）

Step 10 K-means/scripts/k1_closed_pct.py
        → K-means/result/k1_closed_pct.parquet
        → K-means/analysis/k1_closed_pct_dist.png（確認性視覺化）

Step 11 verify/scripts/verify_all.py
        → verify/analysis/v1–v4 圖表
```

## 最終輸出

| 檔案 | 說明 |
|------|------|
| `lifetime kmeans/result/final/removal_final.parquet` | **3,439 名**除名用戶，含 `reason` 欄位 |
| `K-means/result/all_scores.parquet` | 287k 用戶的 K1 餘弦相似度與聚類 |
| `K-means/result/k1_closed_pct.parquet` | K1 Tier-2 用戶的 closed_pct |

## 輔助腳本

| 腳本 | 用途 |
|------|------|
| `K-means/scripts/kmeans_matrix_overview.py` | 產生 K1 聚類總覽圖 |
| `K-means/scripts/sample_reviews.py` | 抽樣異常用戶評論供人工檢視（探索用） |

