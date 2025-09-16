---
title: Week2 Assignment

---


::: info
## Problem 1-1 — 計算 $\nabla a^{[L]}(x)$

**網路前向（forward）**：對 $l=1,\dots,L$，
$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \qquad
a^{[l]} = \sigma^{[l]}(z^{[l]}), \qquad
a^{[0]} = x \in \mathbb{R}^{n_0},
$$
並假設 $n_L=1$。下文中的 $\nabla a^{[L]}(x)$ 一律指 **對輸入 $x$ 的梯度**，記作 $\nabla_{\!x} a^{[L]}(x)$。


### 符號（維度）
- $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}},\; b^{[l]} \in \mathbb{R}^{n_l}$  
- $z^{[l]}, a^{[l]} \in \mathbb{R}^{n_l}$  
- 設 $D^{[l]} := \operatorname{diag}\!\big(\sigma^{[l]\,'}(z^{[l]})\big) \in \mathbb{R}^{n_l \times n_l}$  
- 目標：$\nabla_x a^{[L]}(x) \in \mathbb{R}^{n_0}$

### 想法
使用**反向模式 / VJP**（向量–雅可比乘積）。任一層 $l$ 有
$$
\frac{\partial a^{[L]}}{\partial a^{[l-1]}}
= \Big(\frac{\partial a^{[L]}}{\partial a^{[l]}}\Big)\, D^{[l]} \, W^{[l]} .
$$
因此以「先乘 $D^{[l]}$、再乘 $(W^{[l]})^\top$」的順序自輸出往回傳播，即可把梯度傳回輸入。

### 演算法（先 forward 快取，再 backward VJP）
1. **Forward**：計算並快取 $\{z^{[l]}, a^{[l]}\}_{l=1}^L$。  
2. **Backward**：  
   - 初始化 $g \leftarrow 1$（即 $\partial a^{[L]}/\partial a^{[L]}$，純量）。  
   - 對 $l = L, L\!-\!1, \dots, 1$：  
     - $g \leftarrow D^{[l]} \, g$（元素相乘，作用為 $\sigma^{[l]\,'}(z^{[l]})$）  
     - $g \leftarrow (W^{[l]})^\top g$（把梯度移到 $a^{[l-1]}$）  
   - **輸出** $g$，此時
     $$
     \boxed{\nabla_x a^{[L]}(x) = g \in \mathbb{R}^{n_0}} .
     $$

**緊湊寫法**
$$
\nabla_x a^{[L]}(x)
= (W^{[1]})^\top D^{[1]} (W^{[2]})^\top D^{[2]} \cdots (W^{[L]})^\top D^{[L]} \,\mathbf{1},
$$
其中 $\mathbf{1}\in\mathbb{R}^{n_L}$ 為 1。若輸出層為線性活化，取 $D^{[L]}=I$。  
> 註：**最左邊的因子最後作用**；這是由 $l=L\to1$ 的 backward 掃描得到的 VJP 形式。

### 偽程式碼
```text
# 輸入：x、參數 {W[l], b[l]}、活化函數 σ[l]、層數 L
# forward
a[0] = x
for l = 1..L:
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = σ[l](z[l])

# backward（純量輸出時的 VJP）
g = 1.0
for l = L..1:              # 由後往前
    g = σ'[l](z[l]) ⊙ g    # 元素乘
    g = W[l].T @ g
return g                   # 即 ∇_x a^[L](x)
```
:::



::: info
## Problem 1-2 提問
* 初始化的尺度（$w,b,v$ 的方差）如何影響輸入梯度的大小與穩定性？
* 只用固定學習率時，應如何設計「提早停止」準則，避免過長訓練或過早停止？
* 在不使用 epoch 的前提下，僅用 iteration 設計早停準則有沒有實用的判斷門檻？
* 在報告的可重現性方面，對隨機種子、資料分割與結果波動應提供到什麼程度才算足夠？
:::



::: info
## Problem 2
## Notation

- $x \in [-1,1]$：輸入（純量）。
- $f(x)=\dfrac{1}{1+25x^2}$：目標函數（偶函數）。
- $N_{\text{train}}, N_{\text{valid}}, N_{\text{test}}$：訓練／驗證／測試樣本數。
- **$H$**：隱藏層**寬度**（hidden units 的數量）。本作業中，$H$ 同時也是 **Even-Pair** 單元的**對數**；即共有 $H$ 個「成對」單元。
- 參數與向量維度（皆為長度 $H$ 的向量）：
  - $w=(w_1,\dots,w_H)^\top$：輸入到隱藏層的權重（輸入維度為 1）。
  - $b=(b_1,\dots,b_H)^\top$：隱藏層偏置（非零初始化以破壞奇偶對稱）。
  - $v=(v_1,\dots,v_H)^\top$：隱藏層到輸出的權重。
  - $b_2$：輸出層偏置（純量）。
- $h(x)\in\mathbb{R}^H$：隱藏層特徵向量，其第 $i$ 個分量為 $h_i(x)$（見下）。
- $\hat y(x)$：模型對輸入 $x$ 的預測。
- **iteration**：一次「全批次梯度下降（Batch GD）」的更新步驟。
- 誤差度量

$$
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}\Big(\hat y(x_i)-y_i\Big)^2, \qquad
\mathrm{MaxErr}=\max_{1\le i\le N}\Big|\hat y(x_i)-y_i\Big|.
$$

---

我們希望用一個前饋神經網路逼近

$$
f(x)=\frac{1}{1+25x^2},\qquad x\in[-1,1].
$$

此函數在 $x=0$ 有明顯尖峰、兩側快速衰減，且為**偶函數** $f(x)=f(-x)$。

---

## Method

### Data
- 訓練：從 $[-1,1]$ **均勻抽樣** $N_{\text{train}}=2000$。
- 驗證：同分布抽樣 $N_{\text{valid}}=800$。
- 測試：$[-1,1]$ 上等距 $N_{\text{test}}=1001$（便於作圖與統計）。
- 標籤：$y=f(x)$（不加雜訊）。  
- 隨機種子（data/init） = **7 / 123**。

### Model — Even-Pair MLP（單隱藏層、tanh→linear）
為了讓模型**結構上**滿足偶對稱，對每個隱藏單元使用「正負成對」設計：

- 第 $i$ 個對偶單元的輸出（tanh 逐元素作用）：

$$
h_i(x)=\tanh(w_i x + b_i)\;+\;\tanh(-w_i x + b_i),
$$

因此 $h_i(-x)=h_i(x)$。向量化表示：

$$
h(x)=\tanh(wx+b)\;+\;\tanh(-wx+b)\in\mathbb{R}^H.
$$

- 最終輸出（此處的 **$H$** 即上面定義的隱藏寬度）：

$$
\hat y(x)=b_2+\sum_{i=1}^{H} v_i\,h_i(x)\;=\;b_2+v^\top h(x).
$$

### Loss & Optimization
- 目標函數：$MSE=\frac{1}{N}\sum_{i=1}^N\big(\hat y_i-y_i\big)^2.$

- 最佳化：**全批次梯度下降（Batch GD）**；訓練步數以 **iteration** 記錄；學習率固定。

### Initialization（避免退化成常數解）
- 使用**非零偏置** $b_i\sim\mathcal N(0,0.1^2)$ 破壞奇偶對稱。  
- 輸出層權重初值 $v_i\sim\mathcal N(0,0.5^2)$ **略放大**，避免 $\partial L/\partial w\propto v$ 太小。  
- $w_i\sim\mathcal N(0,2.5^2)$ 提供多樣斜率，讓部分單元在 $x\approx 0$ 具有較大靈敏度。

### Hyperparameters
- 隱藏寬度 $H=64$、learning rate $=3\times10^{-3}$、iterations $=3000$；batch 為全部訓練樣本。

---

## Results

### A. True vs. NN（同圖對照）
![image](https://hackmd.io/_uploads/Byzx0aIigx.png)

模型大致復現中心尖峰與對稱衰減；但可見在 $x\approx0$ 的**尖峰略被低估**、在 $x\approx0.3\sim0.5$ **略高於真值**，顯示存在**輕微的偏差（underfitting）**。

### B. Training/Validation Loss Curves
![image](https://hackmd.io/_uploads/BJI-0aUogg.png)

兩條曲線快速下降並在約 500 次後趨近平穩，且 **validation MSE 幾乎與 training MSE 重合，泛化落差很小**。  
結合 Fig. 1 的形狀觀察，當前模型主要問題是**偏差**（尖峰略矮），而非過擬合。

### C. Errors & Settings Summary

**Errors (on test grid):**
- **Test MSE** = 1.421*10^(-3)
- **Test MaxErr** = 1.115*10^(-1)

**Settings Summary**
| Split / Param | Value |
|---|---:|
| Train / Valid / Test size | 2000 / 800 / 1001 |
| Hidden units (H) | 64 |
| Learning rate | 3*10^(-3) |
| Iterations | 3000 |
| Seeds (data / init) | 7 / 123 |
| **Test MSE** | 1.421*10^(-3) |
| **Test MaxErr** | 1.115*10^(-1) |


---

## Discussion
1. **為何 Even-Pair 有效？** 目標 $f(x)$ 是偶函數；一般 MLP 在對稱資料上，奇函數基底 $\tanh(wx)$ 的梯度會互相抵銷，常導向近似常數或偏歪的解。Even-Pair 先天滿足 $y(x)=y(-x)$，**把不需要的奇成分從函數類別中移除**，使 GD 更容易找到正確的谷底。  
2. **初始化的角色。** 非零 $b_1$ 破壞奇偶對稱；較大的輸出層初值確保 $\partial L/\partial (W_1,b_1)$ 不被抑制，避免長時間只學到常數平均值。  
3. **偏差–變異取捨（結合本結果）.** 目前 generalization gap 極小，但尖峰略低屬偏差：可將 $H$ 提升至 96–128、把 iterations 增至 4000–5000，或將輸出層初值標準差由 0.5 微調至 0.6–0.8；若訓練不穩，可把學習率調為 $2\text{e-}3$。  
4. **普通 MLP 的對照（口述）**：若 $b_1=0$ 或輸出層初值過小，預測常接近水平線或一側下挫，這也解釋了最初「看起來不對」的原因。

---

## Conclusion
在 **GD + MSE** 且以 **iteration** 記錄訓練步數的設定下，我們以**單隱藏層 Even-Pair MLP** 成功逼近 Runge 函數：train/valid 曲線一致下降且幾乎重合（泛化落差小），但仍有輕微偏差（尖峰略矮）。這顯示對稱先驗能顯著改善可訓練性與穩定性，後續可藉由增加容量/步數與微調初始化進一步降低誤差。

---

## Reproducibility
- 環境：Python（Colab CPU 即可），`numpy`、`matplotlib`、`pandas`。  
- 種子：**data/init = 7 / 123**。  
- 產物：`w2_func_vs_nn.png`、`w2_loss_curve.png`、`w2_summary.csv`。  
- 訓練：**全批次 GD**、**MSE**、固定 **iterations**；不使用 Adam，也不使用 “epoch”。
:::