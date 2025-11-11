# Week 10 Assignment

## 1. Probability Flow ODE
Consider the 1D Itô SDE
$$
dx_t = f(x_t,t)\,dt + g(x_t,t)\,dW_t,
$$
whose density $p(x,t)$ satisfies the Fokker–Planck equation
$$
\partial_t p = -\partial_x[f(x,t)p(x,t)] + \tfrac{1}{2}\partial_x^2[g^2(x,t)p(x,t)].
$$
Assume there exists a deterministic flow $dx_t = u(x_t,t)\,dt$ that shares the same marginals as the SDE. Its density must obey the continuity equation
$$
\partial_t p = -\partial_x\big[u(x,t)p(x,t)\big].
$$
Equating the two PDEs gives
$$
-\partial_x[up] = -\partial_x[fp] + \tfrac{1}{2}\partial_x^2[g^2 p].
$$
Integrating over $x$ (with vanishing boundary flux) yields
$$
up = fp - \tfrac{1}{2}\partial_x(g^2 p).
$$
Dividing by $p$ and expanding $\partial_x(g^2 p) = (\partial_x g^2)p + g^2\partial_x p$ gives the velocity field
$$
u(x,t) = f(x,t) - \tfrac{1}{2}\partial_x g^2(x,t) - \tfrac{g^2(x,t)}{2}\partial_x\log p(x,t).
$$
Therefore, the probability-flow ODE is
$$
dx_t = \left[f(x_t,t) - \tfrac{1}{2}\partial_x g^2(x_t,t) - \tfrac{g^2(x_t,t)}{2}\partial_x\log p(x_t,t)\right]dt.
$$

## 2. AI 的未來與機器學習的基石
### 2.1 AI 的未來能力
我設定的遠期目標是打造一個能夠在全球多資產市場中，自主提出、驗證並部署新型量化策略的 AI 研究夥伴。它不僅能同時吸收逐筆交易、鏈上資料、衛星影像、企業供應鏈等異質訊號，還能自動演繹背後的經濟敘事與風險傳導鏈，並以可審計的證據說服監管與投資人。目前的模型最多只能在既有資料上擬合報酬，難以推導出可解釋、合規且可長期維運的策略；而我希望 20 年後的系統能主動發現極端事件前的脆弱結構、提出政策友善的避險方案，減少金融危機對實體經濟與退休金體系的破壞。

### 2.2 涉及的機器學習類型
為了達成這個能力，核心會是多模態表徵的自監督/非監督學習，用來整理龐大且噪聲極高的替代資料；接著以監督式學習訓練可檢驗的預測器（例如條件報酬、流動性缺口、信用擴散），最後透過強化學習在模擬與真實市場的數位分身中做決策，並引入層級式安全約束。資料來源包含交易所逐筆資料、央行與供應鏈公告、API 取得的 ESG/氣候指標與自蒐影像；目標訊號則是策略在多種情境下的風報比、下行風險與解釋用語。強化學習階段需要與市場環境互動並將監管規則視為 reward shaping，以確保策略長期有效且合規。

### 2.3 第一步的模型化
我會鎖定「跨市場系統性風險預警 + 自動對沖設計」的簡化問題：給定美國股票、債券與加密資產的歷史資料與生成式 scenario，AI 必須提出可被人類審計的指標組合，預測 30 天後的系統性壓力，並生成一組可執行的中性投資組合與解釋文字。可測試性來自於回測與代理 sandbox（包含程式化的監管檢核），判斷指標是否提前於 VIX/信用利差發出警示，以及產生的對沖是否在多數 scenario 中降低最大回撤。解決此問題需要變分自編碼器/擴散模型整理異質資料，圖神經網路建模市場關聯，結合分佈式強化學習與凸最佳化來求得約束下的投資解。

（以上 2.1–2.3 合計約 770 字，落在 600–800 字範圍內。）

## 4. Unanswered Questions
- 若擾動係數 $g(x,t)$ 與狀態高度相關，score $\partial_x \log p$ 的估計誤差會如何影響 reverse SDE 的穩定性？是否需要額外修正項以維持熵守恆？
- PF-ODE 與 reverse SDE 在 diffusion model 中都可生成樣本，但兩者對不同噪聲日程的誤差敏感度是否存在理論上的上界？
- 在多維情況且 $G(x,t)$ 不是純量倍數時，Backward SDE 的顯式形式是否仍可寫出？若需切換至 Stratonovich 表示，score-PDE 需要怎樣的修正？
