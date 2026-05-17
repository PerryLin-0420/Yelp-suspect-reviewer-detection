# Yelp 低可信度評論者偵測

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![DuckDB](https://img.shields.io/badge/DuckDB-0.10+-yellow)

針對 [Yelp Open Dataset](https://www.yelp.com/dataset)（1,987,841 名用戶）做的無監督偵測 side project，不需要標記資料。

從兩個維度找行為異常的帳號，再透過店家層次的時間群聚確認是不是真的有協調刷評。本質是一套**行為特徵過濾器**，不是要宣稱抓到了所有假帳號。

---

## 前置：資料庫

分析腳本統一吃 `YELP.duckdb`。需要的資料表：`review`、`user`、`business`、`business_hours`。

從 [Yelp Open Dataset](https://www.yelp.com/dataset) 下載原始 JSON 後，另有一套獨立的清洗腳本負責建表（未包含在本 repo，各實體各自一支 `autobuild_db.bat`，放好 JSON 直接跑）。手動建表或用其他方式匯入也可以，只要 schema 對上即可。

---

## 怎麼跑的

### 分支一：時段 K-Means

每個用戶用 24 維的小時發評比例向量表示。正常人有作息，在不該出現的時段大量發評本身就是信號。

- Yelp 存的是本地時間（實證：PA 用戶發評高峰 = 18:00 美東）
- K-Means（K=20）找出早晨比例異常高的 cluster，再子聚類隔離最極端的 Sub3/Sub4
- 交叉驗證：Sub3/Sub4 的早晨評論，只有 9–14% 落在店家已開業時段（正常群：43%）
- 對全體 287k 用戶算餘弦相似度 → **4,037 名候選人**

### 分支二：帳號生命週期 K-Means

每個用戶的評論在帳號生命週期 [0, 1] 上的密度分布。

- 單評論帳號（n=1,135,945）：28% 在創號後 19 天內發出唯一評論，消失
- 多評論帳號（n=851,896）：找爆發後沉默的模式（低熵、峰值在 0.00）

### 店家群聚確認

光有候選帳號不夠，還要看他們有沒有集中評同一家店。

| 指標 | 定義 |
|------|------|
| `coverage_rate` | 可疑帳號數 / 該店家總評論數 |
| `span_days` | 最早到最晚一筆可疑評論的間距 |
| `dense_30d_pct` | 可疑評論落在最密集 30 天窗口的比例 |
| `suspicion_score` | `coverage_rate × dense_30d_pct` |

7 天內集中 = 協調行為；`suspicion_score ≥ 0.5` = 高重疊。

完整方法論：[ANALYSIS_ZH.md](ANALYSIS_ZH.md) ｜ [ANALYSIS_EN.md](ANALYSIS_EN.md)  
執行順序：[PIPELINE.md](PIPELINE.md)

---

## 結果

| 類別 | 人數 | 佔比 |
|------|------|------|
| 協調刷評（單評論帳號） | 1,424 | 0.072% |
| 協調刷評（爆發型帳號） | 744 | 0.037% |
| 單評論高店家重疊 | 1,193 | 0.060% |
| 時段異常 + 協調確認 | 78 | 0.004% |
| **合計** | **3,439** | **0.173%** |

輸出：`lifetime kmeans/result/final/removal_final.parquet`（`user_id` + `reason`）

---

## 這套方法的特性與局限

**能做到的：**
- 行為異常偵測，不需要標記資料、不審查評論文字
- 兩條路線設計獨立，各自抓不同動機的帳號（見回頭驗證）
- 保守：只有有直接協調證據的帳號才進除名清單

**不保證的：**
- 沒有 precision / recall，無監督方法本來就沒有 ground truth
- 行為「正常」的假帳號抓不到（例如長期潛伏、慢速刷評）
- 同樣的行為模式可能有良性解釋

### 回頭驗證：為什麼兩條路線抓的人動機不同

把 3,439 名 flagged 用戶放回所有維度看，他們在「不是因此被抓」的維度上跟一般用戶幾乎沒有差異（K1 餘弦相似度 0.168 vs 門檻 0.546；closed_pct 差距 +0.018）。

這不是問題——正好說明**兩條路線各自找到了真正不同的行為類別**，不是同一批人的兩種看法：

| 類別 | 行為 | 推測動機 |
|------|------|---------|
| `single_coordinated` | 多帳號 7 天內評同一家店，消失 | 受僱一次性刷評 |
| `burst_coordinated` | 帳號早期爆發後沉默，共享目標店家 | 批量操作或收購帳號 |
| `single_high_overlap` | 所評店家被可疑帳號高度覆蓋 | 刷評生態圈的參與者 |
| `k1_coordinated` | 系統性早晨發評 + 共享目標 | 自動化腳本或遠端操控 |

---

## 專案結構

```
├── README.md
├── PIPELINE.md
├── ANALYSIS_ZH.md / ANALYSIS_EN.md
│
├── user behavier matrix/scripts/   # Step 0: 建 24 維小時矩陣
├── K-means/scripts/                # Steps 1–6, 9–10: 時段分析
├── lifetime kmeans/scripts/        # Steps 7–8: 生命週期分析
└── verify/scripts/                 # Step 11: 回頭驗證
```

---

## 環境

```bash
pip install duckdb pandas numpy scikit-learn matplotlib scipy loguru
```
