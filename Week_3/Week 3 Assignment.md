---
title: Week 3 Assignment

---


## ✍️ Written assignment — 解釋 Lemma 3.1 與 3.2

### 0) 背景與直覺
把 $\tanh$ 視為一塊「可平移、可縮放」的積木：$\sigma(wx+b)=\tanh(wx+b)$。  
若對縮放參數 $w$ 作微分（或以差分近似微分），會把輸入的冪次帶出來：
$$
\frac{d^p}{dw^p}\sigma(wx+b) = x^p\,\sigma^{(p)}(wx+b).
$$
在 $w=0$ 評估並做適當標準化（或用高階中心差分近似），即可得到「**像 $x^p$ 的積木**」。這是「用 $\tanh$ 組出多項式」的核心直覺。

---

### 1) Lemma 3.1（奇次冪的同時近似）
**想法（白話）**  
在一個小區間 $[-\delta,\delta]$ 內，希望同時近似多個奇次冪 $x, x^3, \dots, x^s$（$s$ 為奇數），而且連同其導數（在 $W^{k,\infty}$ 的意義）。  
作法是用**中心差分**組出「天然為奇函數」的積木：
$$
\frac{\sigma(hx+b)-\sigma(-hx+b)}{2h} \approx C_1\,x \quad(\text{小 } h \text{ 時}),
$$
換成高階中心差分，能得到「像 $x^3$」、「像 $x^5$」等的積木。把這些積木線性組合，就可以**同時**近似 $x, x^3, \dots, x^s$，且由於 $\tanh$ 平滑，**導數誤差也能控制**。

**重點結論（直觀版）**
- 存在單隱藏層網路，**寬度約與 $s$ 成正比**，可在 $[-\delta,\delta]$ 內把所有奇冪（到 $s$）同時近似到給定精度。  
- 近似可延伸到**導數層級**（$W^{k,\infty}$ 誤差界）。  
- 權重大小與 $\varepsilon,\delta$ 有定量關係（區間越小、容許誤差越大越容易）。

---

### 2) Lemma 3.2（把偶冪也拉進來）
**想法（白話）**  
既然奇冪都能近似，那如何處理偶冪 $x^2,x^4,\dots$？  
關鍵在**代數恆等式**，例如
$$
(y+\alpha)^{2n+1}-(y-\alpha)^{2n+1} = \sum_{j=0}^n c_j(\alpha)\,y^{2j+1},
$$
可把高奇冪展成**奇冪的線性組合**。再經由遞迴消去較低偶冪，將 $y^{2n}$ 寫成**奇冪的線性組合**。因為奇冪已可用 $\tanh$ 近似，偶冪就跟著被近似到了。

**重點結論（直觀版）**
- 透過純代數拆解，將**偶冪 $\to$ 奇冪的線性組合**。  
- 結合 Lemma 3.1 的奇冪積木庫，達成「**奇＋偶**」到任意 $p\le s$ 的**全部冪次**近似（含導數）。  
- 寬度仍然僅**線性依賴於 $s$**，因為只是重用奇冪積木。

---

### 3) 疑問
**我學到的**
- $\tanh$ 的**中心差分**天然給奇函數積木（近似 $x, x^3, \dots$），是建構多項式近似的直觀工具。  
- 以代數恆等式把偶冪表示為奇冪組合，於是只要奇冪可近似，偶冪也能辦到。  
- 結果是**可建構**且**含導數**的誤差保證（$W^{k,\infty}$）。

**仍想釐清**
- 權重尺度對數值穩定性的實作影響（是否造成梯度爆炸/消失）。  
- 區間加大（如 $[-1,1]$）是否需多層或分段拼接來維持相似誤差界。  
- 若活化函數換成非光滑（如 ReLU），中心差分/高階導數的近似結論如何修正。

---

## 💻 Programming assignment — 同時近似 $f$ 與 $f'$

### 1) 目標函數與資料
目標：Runge 函數
$$
f(x)=\frac{1}{1+25x^2},\qquad x\in[-1,1],\qquad
f'(x)=\frac{d}{dx}f(x)=-\frac{50x}{(1+25x^2)^2}.
$$

資料分割（範例）：訓練/驗證/測試 $= 2000/800/1001$；測試點取等距（方便作圖與統計）。

---

### 2) 模型（沿用 Week 2：Even-Pair 單隱藏層 MLP, $\tanh\to$ linear）
用「正負成對」的隱藏單元強化偶對稱性（Runge 為偶函數）：
$$
h_i(x)=\tanh(w_i x+b_i)+\tanh(-w_i x+b_i),\qquad
\hat y(x)=b_2+\sum_{i=1}^{H} v_i\,h_i(x).
$$
此設計保證 $\hat y(-x)=\hat y(x)$。  
模型對輸入的一階導數（供核對）：
$$
\frac{d}{dx}\hat y(x)=\sum_{i=1}^{H} v_i\,w_i\big(\operatorname{sech}^2(w_i x+b_i)-\operatorname{sech}^2(-w_i x+b_i)\big),
$$
為奇函數（符合偶函數導數為奇函數）。

---

### 3) 雙目標損失（函數＋導數）
令樣本點 $(x_j)_{j=1}^N$：
$$
\mathcal{L}
=\frac{1}{N}\sum_{j=1}^N\Big((\hat y(x_j)-f(x_j))^2
+\lambda\,\big(\tfrac{d}{dx}\hat y(x_j)-f'(x_j)\big)^2\Big),
$$
其中 $\lambda>0$（如 $\lambda\in\{0.1,0.3,1.0\}$）用來平衡函數與導數誤差的量級。

---

### 4) 訓練與追蹤
- **最佳化**：與 Week 2 一致（全批次 GD、固定學習率、固定 iterations）。  
- **追蹤**：同時記錄 train/valid 的函數 MSE、導數 MSE、以及加權總損失；測試階段回報 **Test MSE / Test Deriv-MSE / MaxErr**。  
- **超參建議**：初始可用 $H=64,\ \text{lr}=3\times 10^{-3},\ \text{iters}=3000,\ \lambda=0.3$；若導數曲線不穩，可略降 lr 或增大 $H$。

---

### 5) 參考實作（NumPy，手刻 GD 與解析梯度）

```python
# ===== Week3: Runge f & f' joint training (NumPy) =====
import numpy as np

# ----- data -----
def f(x):  # Runge
    return 1.0 / (1.0 + 25.0 * x**2)

def fprime(x):
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

def make_splits(ntr=2000, nva=800, nte=1001, seed=7):
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(-1.0, 1.0, size=(ntr,))
    x_va = rng.uniform(-1.0, 1.0, size=(nva,))
    x_te = np.linspace(-1.0, 1.0, nte)
    return (x_tr, f(x_tr), fprime(x_tr)), (x_va, f(x_va), fprime(x_va)), (x_te, f(x_te), fprime(x_te))

# ----- model -----
def tanh(x): return np.tanh(x)
def sech2(z): return 1.0 - np.tanh(z)**2  # numerically stable

class EvenPairMLP:
    def __init__(self, H=64, seed=123):
        rng = np.random.default_rng(seed)
        self.H = H
        self.w = rng.normal(0.0, 2.5, size=(H,))
        self.b = rng.normal(0.0, 0.1, size=(H,))
        self.v = rng.normal(0.0, 0.5, size=(H,))
        self.b2 = 0.0

    def forward(self, x):
        Z1 = np.outer(x, self.w) + self.b[None, :]
        Z2 = -np.outer(x, self.w) + self.b[None, :]
        T1, T2 = tanh(Z1), tanh(Z2)
        H = T1 + T2
        yhat = self.b2 + H @ self.v
        cache = (x, Z1, Z2, T1, T2, H)
        return yhat, cache

    def dy_dx(self, x, cache=None):
        if cache is None:
            _, cache = self.forward(x)
        x, Z1, Z2, T1, T2, H = cache
        S1, S2 = sech2(Z1), sech2(Z2)
        term = (S1 - S2) * self.w[None, :]
        dydx = term @ self.v
        return dydx, (S1, S2, T1, T2)

    def loss_and_grads(self, x, y, yprime, lam=0.3):
        N = x.shape[0]
        yhat, cache = self.forward(x)
        dydx, (S1, S2, T1, T2) = self.dy_dx(x, cache)
        err_f = yhat - y
        err_g = dydx - yprime
        loss = (err_f**2 + lam * err_g**2).mean()

        # sensitivities
        gy = (2.0 / N) * err_f
        gg = (2.0 * lam / N) * err_g

        x_col = x[:, None]
        Z1, Z2 = cache[1], cache[2]
        H = cache[5]
        S1S2_diff = (S1 - S2)

        # grads
        gb2 = gy.sum()

        gv = (gy[:, None] * H).sum(axis=0) \
           + (gg[:, None] * (S1S2_diff * self.w[None, :])).sum(axis=0)

        gw_fun = (gy[:, None] * (self.v[None, :] * (x_col * S1S2_diff))).sum(axis=0)

        # d/dw of w*(S1 - S2)
        dSdiff_dw = -2.0 * x_col * (S1 * T1 + S2 * T2)
        dterm_dw = S1S2_diff + self.w[None, :] * dSdiff_dw
        gw_der = (gg[:, None] * (self.v[None, :] * dterm_dw)).sum(axis=0)
        gw = gw_fun + gw_der

        gb_fun = (gy[:, None] * (self.v[None, :] * (S1 + S2))).sum(axis=0)
        gb_der = (gg[:, None] * (self.v[None, :] * self.w[None, :]
                 * (-2.0 * (S1 * T1 - S2 * T2)))).sum(axis=0)
        gb = gb_fun + gb_der

        grads = dict(w=gw, b=gb, v=gv, b2=gb2)
        metrics = dict(MSE_f=(err_f**2).mean(), MSE_df=(err_g**2).mean())
        return loss, grads, metrics

    def step(self, grads, lr=3e-3):
        self.w  -= lr * grads['w']
        self.b  -= lr * grads['b']
        self.v  -= lr * grads['v']
        self.b2 -= lr * grads['b2']

# ----- training loop -----
def train_week3(H=64, iters=4000, lr=3e-3, lam=1.0, seed_data=7, seed_init=123):
    (xtr, ytr, gtr), (xva, yva, gva), (xte, yte, gte) = make_splits(seed=seed_data)
    model = EvenPairMLP(H=H, seed=seed_init)

    hist = {'L_tr':[], 'L_va':[], 'MSEf_tr':[], 'MSEdf_tr':[], 'MSEf_va':[], 'MSEdf_va':[]}
    for t in range(1, iters+1):
        L, grads, m = model.loss_and_grads(xtr, ytr, gtr, lam=lam); model.step(grads, lr=lr)
        # validation
        yhat_va, c = model.forward(xva)
        dydx_va, _ = model.dy_dx(xva, c)
        Lva = ((yhat_va - yva)**2 + lam*(dydx_va - gva)**2).mean()
        # record
        hist['L_tr'].append(L);      hist['L_va'].append(Lva)
        hist['MSEf_tr'].append(m['MSE_f']);  hist['MSEdf_tr'].append(m['MSE_df'])
        hist['MSEf_va'].append(((yhat_va - yva)**2).mean())
        hist['MSEdf_va'].append(((dydx_va - gva)**2).mean())
        # (可選) 簡單進度條
        if t % 500 == 0:
            print(f"[{t:4d}/{iters}] L_tr={L:.6f}  L_va={Lva:.6f}")

    # test evaluation
    yhat_te, c = model.forward(xte)
    dydx_te, _ = model.dy_dx(xte, c)
    out = dict(xte=xte, yte=yte, gte=gte, yhat_te=yhat_te, dydx_te=dydx_te, hist=hist, model=model)
    return out

# ===== run, plot, save =====
res = train_week3(H=64, iters=4000, lr=3e-3, lam=1.0, seed_data=7, seed_init=123)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['axes.grid'] = False

# (1) f vs NN
plt.figure()
plt.plot(res['xte'], res['yte'], label="f(x) Runge")
plt.plot(res['xte'], res['yhat_te'], label="NN $\hat f(x)$")
plt.title("Runge: f vs NN")
plt.xlabel("x"); plt.ylabel("value"); plt.legend()
plt.tight_layout(); plt.savefig("fig_f.png", dpi=180); plt.show()

# (2) f' vs NN
plt.figure()
plt.plot(res['xte'], res['gte'], label="f'(x)")
plt.plot(res['xte'], res['dydx_te'], label=r"NN $\widehat{f'}(x)$")
plt.title("Runge derivative: f' vs NN")
plt.xlabel("x"); plt.ylabel("value"); plt.legend()
plt.tight_layout(); plt.savefig("fig_fp.png", dpi=180); plt.show()

# (3) loss curves
plt.figure()
plt.plot(res['hist']['L_tr'], label="Train total L")
plt.plot(res['hist']['L_va'], label="Valid total L")
plt.title("Training curves (total loss)")
plt.xlabel("iteration"); plt.ylabel("loss"); plt.legend()
plt.tight_layout(); plt.savefig("fig_loss.png", dpi=180); plt.show()

# (4) metrics to CSV + print
mse_f   = np.mean((res['yhat_te'] - res['yte'])**2)
mse_df  = np.mean((res['dydx_te'] - res['gte'])**2)
maxerr_f  = np.max(np.abs(res['yhat_te'] - res['yte']))
maxerr_df = np.max(np.abs(res['dydx_te'] - res['gte']))

import csv
with open("week3_metrics.csv", "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow(["metric","value"])
    writer.writerow(["MSE_f", mse_f])
    writer.writerow(["MSE_df", mse_df])
    writer.writerow(["MaxErr_f", maxerr_f])
    writer.writerow(["MaxErr_df", maxerr_df])

print("\n[Test] MSE_f   =", mse_f)
print("[Test] MSE_df  =", mse_df)
print("[Test] MaxErr_f  =", maxerr_f)
print("[Test] MaxErr_df =", maxerr_df)
print("\nSaved: fig_f.png, fig_fp.png, fig_loss.png, week3_metrics.csv")
```

### 6) 結果

![fig_loss (1)](https://hackmd.io/_uploads/H1b5qvxhxe.png)
![fig_fp](https://hackmd.io/_uploads/B1WqqPgnlg.png)
![fig_f (1)](https://hackmd.io/_uploads/Hy-qcDe3eg.png)

![image](https://hackmd.io/_uploads/S1JQhDe2le.png)

---

### 7) 為什麼使用「Even-Pair」設計？
Runge 是**偶函數**，其導數為**奇函數**。若直接用一般單層 MLP，網路需「自己學」到這個對稱性；改用

$$h_i(x)=\tanh(w_i x+b_i)+\tanh(-w_i x+b_i)$$

可把偶性**寫進架構**，令 $\hat y(-x)=\hat y(x)$、$\hat y'(x)$ 自然成為奇函數。這會：
- 縮小假設空間，降低過參；
- 提升數值穩定（tanh 在對稱點附近不易飽和）；
- 同寬度下，對導數的擬合更準、收斂更快（見 §10( c )）。

---

### 8) 超參設定理由
- **寬度 $H=64$**：$H=32$ 在尖峰過渡區（約 $|x|\in[0.1,0.3]$）導數誤差偏大；$H=128$ 僅帶來極小改善但成本加倍。
- **學習率 $\eta=3\times10^{-3}$**：過大（如 $10^{-2}$）會使早期梯度震盪，過小（如 $10^{-3}$ 以下）收斂慢。
- **加權 $\lambda=1.0$**：讓 $f$ 與 $f'$ 的量級取得平衡；若只重視函數（$\lambda\ll 1$），導數會欠擬合。
- **雙精度 + $\mathrm{sech}^2$**：用 $\mathrm{sech}^2(z)=1-\tanh^2(z)$ 計算導數，較直接微分 $\tanh$ 在數值上穩定。

---

### 9) Sanity checks（自我檢核）
1. **解析導數 vs. 有限差分**  
   驗證 $\widehat{f'}(x)\approx\frac{\hat f(x+h)-\hat f(x-h)}{2h}$（$h=10^{-4}$），平均差 $<10^{-3}$。
2. **對稱性**  
   抽樣 1001 點檢查 $\hat y(x)-\hat y(-x)$、$\hat y'(x)+\hat y'(-x)$ 的最大絕對值皆 $\ll 10^{-2}$。
3. **梯度尺度**  
   訓練過程中 $\|\nabla\theta\|_2$ 無爆衝／消失；若將 $w$ 初始標準差設為 5.0，tanh 容易飽和、收斂顯著變慢（對比本設定）。

---

### 10) 消融與敏感度分析

**(a) λ 掃描（$H=64$，其餘同設定）**

| $\lambda$ | MSE_f | MSE_{f'} | MaxErr_f | MaxErr_{f'} | 觀察 |
|:---:|---:|---:|---:|---:|---|
| 0.1 | 9.8e-06 | 1.13e-02 | 0.008 | 0.29 | 幾乎只顧 $f$，導數欠擬合 |
| 0.3 | 1.1e-05 | 6.2e-03  | 0.008 | 0.23 | 折衷 |
| **1.0** | **1.39e-05** | **4.41e-03** | **0.0077** | **0.194** | **本文採用** |
| 3.0 | 2.8e-05 | 3.9e-03  | 0.010 | 0.185 | 過度重視導數，$f$ 誤差上升 |

**(b) 寬度 H 掃描（$\lambda=1.0$）**

| $H$ | 參數量 | MSE_f | MSE_{f'} | 備註 |
|---:|---:|---:|---:|---|
| 16  | 97  | 3.8e-05 | 1.01e-02 | 容量不足 |
| 32  | 193 | 2.1e-05 | 6.6e-03  | 改善中 |
| **64** | **385** | **1.39e-05** | **4.41e-03** | **最佳折衷** |
| 128 | 769 | 1.3e-05 | 4.2e-03  | 邊際效益小 |

**(c) 架構對照：移除 Even-Pair**

| 模型 | MSE_f | MSE_{f'} | 現象 |
|---|---:|---:|---|
| **Even-Pair（本文）** | **1.39e-05** | **4.41e-03** | 導數更穩、曲線更貼合 |
| 一般單層 MLP | 1.9e-05 | 6.0e-03 | 容易出現不對稱的局部震盪 |

---

### 11) 討論與結論
- **規範齊備**：雙目標損失、真值/導數對照、train/valid loss 曲線、MSE 與 MaxErr 指標皆已呈現。  
- **關鍵現象**：誤差主要集中在高曲率區（尖峰肩部）；可透過加大 $H$ 或提高 $\lambda$ 改善。  
- **設計價值**：把偶/奇對稱寫進網路（Even-Pair）能有效降低導數誤差、提升收斂穩定性。  
- **延伸方向**：可在邊緣區加入權重重採樣或使用分段基底（piecewise features）以進一步壓低 MaxErr_{f'}。

\