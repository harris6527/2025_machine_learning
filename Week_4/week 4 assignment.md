---
title: week 4 assignment

---


# Written Assignment

1) 當資料線性可分且未加正則化時，為何 logistic 的參數範數 ‖w‖→∞？其方向是否收斂到**最大間隔**分隔超平面？
3) 在哪些條件（樣本數 n、維度 d、條件數）下 IRLS 會優於一階法（SGD/Adam）？何時反而成本高、表現不佳？
4) 在類別極不平衡時，「對 loss 重新加權」與「改變決策閾值」何時等價、何時不等價？
6) 若誤判成本不對稱且不確定，有哪些**具原理**的方式設定分類閾值？
11) 除了 Accuracy、F1，應該回報哪些**決策時**指標（如 Precision@k、期望成本）以貼近實際使用情境？
12) 當 \(X^\top X\) 近奇異時，哪種解法（QR、SVD、Cholesky）較合適？理由是什麼？
13) 在 LWLR 中，帶寬 τ 應如何選擇以平衡偏差–變異，尤其在**地理邊界**（鄰居稀少）時？
14) 當局部權重導致 \(X^\top W X\) 近奇異時，需採取哪些防護（權重下限截斷、加入 ridge、數值穩定技巧）？
17) 隨機切分 train/test 會否造成**空間洩漏**（地理相鄰點高度相依）？若改用**區塊式空間交叉驗證**，誤差估計會如何改變？
18) 在本網格下，哪些簡單地理特徵（海拔、距海距離、海陸遮罩）最可能提升表現？如何量化各特徵的 ΔR²？

20) 哪些隨機來源（資料切分、求解器初始化、洗牌）會顯著影響結果？為了可重現性，應固定/回報哪些項目？



# Programming Assignment

> **任務**：由中央氣象署 O-A0038-003（小時溫度觀測分析格點）建立兩個資料集：  
> (1) 分類（有效/無效），(2) 回歸（溫度值），並各訓練一個模型與評估。

---

## 1. 資料來源與規格

- 產品：**O-A0038-003**（小時溫度觀測分析格點），單位：°C  
- 網格：**67 × 120**  
- 左下角座標：**(120.00E, 21.88N)**  
- 解析度：Δlon=Δlat=**0.03°** → lon 120.00→121.98（67 個點），lat 21.88→25.45（120 個點）  
- 無效值：**-999**（以 label=0 表示；有效為 label=1）  
- 值序：**先沿經度 67 點，再沿緯度 120 列**（共 8040 個值，以逗號分隔，含科學記號）  
- 原始 XML 結構：有 **default namespace**，數值位於 `<dataset>/<Resource>/<Content>` 的長字串中

---

## 2. 前處理流程

### 2.1 XML 解析（Namespace-aware XPath）
- 使用 `lxml.etree` 解析，命名空間對應：`ns = {'c': 'urn:cwa:gov:tw:cwacommon:0.1'}`  
- 取值：`//c:dataset/c:Resource/c:Content/text()` → 取得一串以逗號分隔的數字文字  
- 以正則擷取全部浮點數（含科學記號），長度應為 **8040**

### 2.2 形狀與座標
- 將 8040 個值 `reshape` 為 `(NY=120, NX=67)`；  
- 建立座標：  
  `lons = 120.00 + 0.03 * arange(67)`，  
  `lats = 21.88 + 0.03 * arange(120)`；  
- 以 `meshgrid(lons, lats)` 形成 `(NY, NX)`，再 `ravel()` 與數值對齊，攤平成表格

### 2.3 兩份資料集
- **分類集**：`(lon, lat, label)`，其中 `label = 1`（value ≠ -999），`0`（value = -999）  
- **回歸集**：`(lon, lat, value)`，**僅保留**非 -999 的有效值

---

## 3. 模型與設定

### 3.1 分類（Logistic Regression）
- 特徵：`(lon, lat)`  
- 分割：`train/test = 80/20, random_state=42, stratify=label`  
- 管線：`StandardScaler → LogisticRegression(max_iter=1000, class_weight='balanced')`  
  - `class_weight='balanced'` 用於對抗標籤不平衡

### 3.2 回歸（多項式線性回歸）
- 特徵：`(lon, lat)`  
- 分割：`train/test = 80/20, random_state=42`  
- 管線：`PolynomialFeatures(degree=3, include_bias=False) → LinearRegression`  
  - 可將 `degree=2/3` 比較，並可改用 `Ridge` 抑制過擬合  
  - **不使用**尚未授課的模型（如 KNN）亦可達到平滑曲面擬合

---

## 4. 實驗結果（Test Set）

### 4.1 檔案產出
- `week4_classification.csv`（8040 列；欄位：lon, lat, label）  
- `week4_regression.csv`（3495 列；欄位：lon, lat, value；**不含 -999**）  
- `week4_cls_metrics.csv`、`week4_reg_metrics.csv`（指標摘要）  
- `week4_reg_sample_preds.csv`（回歸測試集 20 筆對照）

### 4.2 指標
**分類（Logistic Regression）**  
- Accuracy = **0.5740**  
- Precision = **0.5086**  
- Recall = **0.5951**  
- F1 = **0.5485**  

**回歸（Polynomial Linear Regression, degree=3）**  
- MAE = **3.0334**  
- RMSE = **4.4450**  
- R² = **0.4176**  

> 回歸值範圍：**[-1.90, 30.00] °C**  
> 標籤分佈：分類 0: **4545**、1: **3495**（與回歸筆數一致）

---

## 5. 討論與觀察

1. **標籤不平衡**：海域/缺測區造成大量無效點（label=0），`class_weight='balanced'` 有助於提升 Recall 與 F1。  
2. **僅用座標的限制**：R² ≈ 0.42 顯示空間平滑性存在，但地形（海拔）、海陸效應、地方性天氣現象等未納入，故上限受限。  
3. **模型選擇**：回歸用多項式線性回歸（課內範圍），比線性平面更能擬合彎曲的空間曲面；`degree` 增加會提昇擬合但也可能過擬合。  
4. **數值 sanity check**：值域 -1.9～30 °C 合理；67×120 網格方向與步距 0.03° 經驗證無誤。

---

## 6. 可能改進（未實作）

- **特徵工程**：加入海拔（DEM）、距海距離、地形分類、方位/坡度等；  
- **空間方法**：IDW/Kriging 或以鄰近點做更合理的插值；  
- **模型比較**：Ridge/Lasso、樹模型（RandomForest/GBDT）、高斯過程回歸；  
- **時間維度**：若能取得連續時刻，加入 hour/day-of-year 等週期特徵。

---

## 7. 可重現與提交

- 主要輸出：`week4_classification.csv`, `week4_regression.csv`, `week4_cls_metrics.csv`, `week4_reg_metrics.csv`, `week4_reg_sample_preds.csv`  
- 建議附上：`README.md`（含環境版本）與一張 67×120 的熱度圖（檢核 reshape 方向）  
- 版本資訊（建議）：Python、scikit-learn 版本（可由 `pip freeze` 或 `sklearn.__version__` 記錄）

---

## 附錄 A：Namespace-aware XPath 片段

```python
from lxml import etree
ns = {'c': 'urn:cwa:gov:tw:cwacommon:0.1'}
tree = etree.parse(DATA_PATH)
content_str = tree.xpath('//c:dataset/c:Resource/c:Content/text()', namespaces=ns)[0]
```

## 附錄 B：熱度圖（驗證 67×120 方向）

```python
import matplotlib.pyplot as plt
vals_grid = df_full.pivot_table(index='lat', columns='lon', values='value').values
plt.figure(figsize=(5,9))
plt.imshow(vals_grid, origin='lower', aspect='auto')
plt.title('Temperature Grid (67×120)')
plt.colorbar(label='°C')
plt.xlabel('lon index (0..66)'); plt.ylabel('lat index (0..119)')
plt.tight_layout(); plt.show()
```
