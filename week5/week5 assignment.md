---
title: week5 assignment

---

### 1. Multivariate normal pdf integrates to 1

Given
$$
f(x)=\frac{1}{\sqrt{(2\pi)^k\det(\Sigma)}}\,
\exp\!\left(-\tfrac12 (x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right),
\qquad x,\mu\in\mathbb{R}^k,\ \Sigma\succ 0,
$$
show that $\int_{\mathbb{R}^k} f(x)\,dx=1$.

**Proof.** Let $y=\Sigma^{-1/2}(x-\mu)$ where $\Sigma^{1/2}$ is the (symmetric) matrix square root of $\Sigma$. Then $x=\mu+\Sigma^{1/2}y$ and $dx = \lvert\Sigma^{1/2}\rvert\,dy = \lvert\Sigma\rvert^{1/2}\,dy$. Also
$$
(x-\mu)^\top \Sigma^{-1}(x-\mu) = y^\top y = \|y\|^2 .
$$
Hence
\[
\begin{aligned}
\int_{\mathbb{R}^k} f(x)\,dx
&= \frac{1}{\sqrt{(2\pi)^k\lvert\Sigma\rvert}}
   \int_{\mathbb{R}^k}\exp\!\left(-\tfrac12\|y\|^2\right)\,\lvert\Sigma\rvert^{1/2}\,dy \\
&= \frac{1}{(2\pi)^{k/2}}\int_{\mathbb{R}^k} e^{-\|y\|^2/2}\,dy .
\end{aligned}
\]
Because the integrand factorizes across coordinates,
$$
\int_{\mathbb{R}^k} e^{-\|y\|^2/2}\,dy
= \prod_{i=1}^k \int_{-\infty}^{\infty} e^{-y_i^2/2}\,dy_i
= (2\pi)^{k/2}.
$$
Therefore the product with $(2\pi)^{-k/2}$ equals $1$. $\square$

---

### 2. Matrix calculus / trace identities

Let $A,B\in\mathbb{R}^{n\times n}$ and $x\in\mathbb{R}^n$.

#### ( a ) Gradient of $\mathrm{tr}(AB)$ w.r.t. $A$
Using differentials and $\mathrm{tr}(X^\top Y)=\langle X,Y\rangle_F$:
$$
d\,\mathrm{tr}(AB)=\mathrm{tr}(dA\,B)=\mathrm{tr}(B^\top dA)
=\langle B^\top,\, dA\rangle_F
\ \Rightarrow\ 
\frac{\partial}{\partial A}\,\mathrm{tr}(AB)=B^\top .
$$

#### ( b ) Quadratic form as a trace
$$
x^\top A x
= \mathrm{tr}(x^\top A x)
= \mathrm{tr}(A x x^\top)
= \mathrm{tr}(x x^\top A).
$$


#### ( c ) Multivariate Gaussian — Maximum Likelihood Estimators

Assume $x_1,\dots,x_n \stackrel{\text{iid}}{\sim}\mathcal{N}_k(\mu,\Sigma)$ with $\Sigma\succ 0$.
The log-likelihood is
$$
\ell(\mu,\Sigma)= -\frac{nk}{2}\log(2\pi) - \frac{n}{2}\log\det(\Sigma)
-\frac12 \sum_{i=1}^n (x_i-\mu)^\top \Sigma^{-1} (x_i-\mu).
$$

**Estimator of $\mu$.** Differentiate and set to zero:
$$
\frac{\partial \ell}{\partial \mu}
= \Sigma^{-1}\sum_{i=1}^n (x_i-\mu) = 0
\ \Rightarrow\ 
\widehat{\mu} = \bar{x} := \frac{1}{n}\sum_{i=1}^n x_i.
$$

**Estimator of $\Sigma$.** Substitute $\widehat{\mu}=\bar{x}$ and let
$$
S = \frac{1}{n}\sum_{i=1}^n (x_i-\bar{x})(x_i-\bar{x})^\top .
$$
Using $\partial \log\det(\Sigma)/\partial\Sigma = (\Sigma^{-1})^\top$ and
$\partial\,\mathrm{tr}(\Sigma^{-1}C)/\partial\Sigma = -(\Sigma^{-1} C\,\Sigma^{-1})^\top$, the score equation gives
$$
-\frac{n}{2}\Sigma^{-1} + \frac{1}{2}\Sigma^{-1}\!\left(\sum_{i=1}^n (x_i-\bar{x})(x_i-\bar{x})^\top\right)\!\Sigma^{-1}=0
\ \Rightarrow\ 
\widehat{\Sigma} = \frac{1}{n}\sum_{i=1}^n (x_i-\bar{x})(x_i-\bar{x})^\top = S .
$$

> Note: This is the **MLE** (uses $1/n$). The unbiased sample covariance uses $1/(n-1)$.

---

### 3. Unanswered Questions

老師好～我想做一個「買方量化／程式交易」方向的期末專題，但我知道本課程重點是機器學習本身、不是金融。想請教幾個跟 ML 評估與作業規範相關的問題：

1) 題目範圍：只要重點在 ML 方法與實驗設計，題材用金融可以嗎？
2) 資料來源：可用公開免費資料（Yahoo/FinMind）嗎？是否需要先給您看資料描述與欄位？
3) 監督目標：您比較建議做分類（上/下）、回歸（未來報酬）、還是序列預測（多步）？
4) 分割方式：時間序列要用「訓練在前、測試在後」即可嗎？需要做 rolling / walk-forward 嗎？
6) 穩健性：希望看到哪些穩健性測試？（不同時間段、打亂標籤、特徵移除 ablation）
7) 指標與報表：您最想看的核心指標是哪些？（例如 MAE/MSE、AUC、F1、時間序列的 MAPE、RMSE）
8) 可重現性：是否要求固定 random seed、提交 requirements.txt、以及結果可重跑的腳本？
9) 複雜度上限：模型複雜度有建議的上限嗎？是否需要先從基準模型做起？
10) 特徵工程：對時序資料是否建議只用「過去可得」的滯後特徵？能否使用技術指標作為工程示例？
13) 報告重點：結果一般但設計嚴謹（資料清理、分割、檢核完整），是否仍可拿到不錯的分數？
14) 原始碼規範：是否需要 notebook + 模組化程式結構、以及說明檔 README？
15) 期限與里程碑：期中前需要給您題目與資料示例嗎？是否需要期中進度簡報？


