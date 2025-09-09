---
title: Week 1 Assignment

---

# Week 1 Assignment


## (1) 一步 SGD 的參數更新（只代入、不需化簡）

模型：
$$
h(x_1,x_2)=\sigma(z),\qquad z=b+w_1x_1+w_2x_2.
$$

單筆資料 $(x_1,x_2,y)=(1,2,3)$，**初始參數** $(b,w_1,w_2)=(4,5,6)$。

先代入資料：
$$
z=4+5\cdot1+6\cdot2=21,\qquad h=\sigma(21).
$$

**計算 $\nabla_{\theta}h$（chain rule） :**
$$
\frac{\partial h}{\partial z}=\sigma'(z)=\sigma(z)\big(1-\sigma(z)\big),
$$
$$
\frac{\partial h}{\partial b}=\frac{\partial h}{\partial z}\frac{\partial z}{\partial b}=\sigma'(z)\cdot 1,\quad
\frac{\partial h}{\partial w_1}=\frac{\partial h}{\partial z}\frac{\partial z}{\partial w_1}=\sigma'(z)\cdot x_1,\quad
\frac{\partial h}{\partial w_2}=\frac{\partial h}{\partial z}\frac{\partial z}{\partial w_2}=\sigma'(z)\cdot x_2.
$$
因此
$$
\nabla_{\theta}h=(\frac{\partial h}{\partial b},\frac{\partial h}{\partial w_1},\frac{\partial h}{\partial w_2})=\sigma(z)\big(1-\sigma(z)\big)\,[\,1,\ x_1,\ x_2\,].
$$
帶入本題 $(x_1,x_2)=(1,2)$、$z=21$：
$$
\nabla_{\theta}h = \sigma(21)\big(1-\sigma(21)\big)\,[\,1,\ 1,\ 2\,].
$$

**所以，由 SGD 一步更新 :**
$$
\theta^{\text{1}}
=\theta^{\text{0}}+2\alpha\,(y-h)\,\nabla_{\theta}h
$$
$$
= [4,\ 5,\ 6] + 2\alpha\big(3-\sigma(21)\big)\sigma(21)\big(1-\sigma(21)\big)\,[\,1,\ 1,\ 2\,]
$$
$$
=\begin{bmatrix}
\ 4 + 2\alpha\,\big(3-\sigma(21)\big)\,\sigma(21)\big(1-\sigma(21)\big) 
\\ \ 5 + 2\alpha\,\big(3-\sigma(21)\big)\,\sigma(21)\big(1-\sigma(21)\big)\cdot 1 \ 
\\ \ 6 + 2\alpha\,\big(3-\sigma(21)\big)\,\sigma(21)\big(1-\sigma(21)\big)\cdot 2 \ 
\end{bmatrix}^T
$$


## (2) Sigmoid 的高階導數與雙曲函數關係

記號：$s=\sigma(x)=\dfrac{1}{1+e^{-x}}$，$s'=\dfrac{ds}{dx}$，$s''=\dfrac{d^2s}{dx^2}$，依此類推。

### (a) $\sigma^{(k)}(x)$, $k=1,2,3$

**Step 1：先由定義直接求一次導數並化成 $s(1-s)$**

$$
\sigma(x)=\big(1+e^{-x}\big)^{-1}
\quad\Longrightarrow\quad
\sigma'(x)=-\big(1+e^{-x}\big)^{-2}\cdot(-e^{-x})
=\frac{e^{-x}}{(1+e^{-x})^2}.
$$

又

$$
s=\frac{1}{1+e^{-x}},\qquad
1-s=\frac{e^{-x}}{1+e^{-x}}
\ \Longrightarrow\
s(1-s)=\frac{e^{-x}}{(1+e^{-x})^2}.
$$

故

$$
\boxed{\ \sigma'(x)=s(1-s)\ }.
$$

**Step 2：用 $s' = s - s^2$ 求二階導數**

$$
s'=\sigma'(x)=s(1-s)=s-s^2.
$$

兩邊對 $x$ 微分：

$$
s''=s'-2ss'=(1-2s)\,s'.
$$

代回 $s'=s(1-s)$ 得

$$
\boxed{\ \sigma''(x)=s(1-s)(1-2s)\ }.
$$

**Step 3：用 $s''=s'(1-2s)$ 求三階導數**

先把二階寫成乘積方便套乘法則：

$$
s''=s'(1-2s).
$$

微分得

$$
s'''=s''(1-2s)+s'(-2s')=s''(1-2s)-2(s')^2
$$

$$
=s(1-s)(1-2s)-2(s(1-s))^2=1-6s+6s^2.
$$

因此

$$
\boxed{\ \sigma'''(x)=s(1-s)\big(1-6s+6s^2\big)\ }.
$$



### (b) 與雙曲正切 $\tanh$ 的關係


$$
\sigma(x)=\frac{1}{1+e^{-x}}
=\frac{e^{x/2}}{e^{x/2}+e^{-x/2}}=\frac{2e^{x/2}}{2\,(e^{x/2}+e^{-x/2})}
$$


$$
=\frac{\frac{e^{x/2}+e^{-x/2}+e^{x/2}-e^{-x/2}}{e^{x/2}+e^{-x/2}}}{2}=\frac{1+\frac{e^{x/2}-e^{-x/2}}{e^{x/2}+e^{-x/2}}}{2}=\frac{1+\tanh\!\left(\tfrac{x}{2}\right)}{2}.
$$



## (3) 課後延伸思考
  
- **學習率**怎麼選？目前有哪些熱門或常見的選法？  
- 若 $f$ 並非處處可微，或導數計算很複雜時，有沒有替代優化方法及其適用情境?
- 初值使 $z$ 取得過大正/負值時，$\sigma'(z)\to 0$，會遇到**梯度消失**；有哪些簡單對策（初始化、特徵縮放、換激活）？  
- **Mini-batch** 與 **SGD** 在收斂穩定性與效率上的取捨是什麼？何時選擇較大的 batch？
