# Week 11 Assignment

## 1. Responsible-AI reflections on Weeks 1-10
題目要求把 Week_1~10 累積尚未解答的疑惑逐一整理，並確認是否已有研究可支撐。以下依照我在期末專題（跨資產風控協作中心）準備時保留下來的疑慮，以「原始問題 → 現有研究或 open problem → 如何被期末專題採用」的格式呈現，並提供可查證的文獻與連結。

### 問題 1（Week 1-2：資料來源與 SGD 前處理）
*原始疑問*：如何在多來源金融/ESG 資料串流中，自動生成可供監理查核的資料說明與權限證明，以免 SGD 在未授權資料上更新？  
*研究狀態*：已有研究。Gebru 等人提出的 Datasheets for Datasets 以及 Mitchell 等人的 Model Cards for Model Reporting，分別說明要如何記錄資料集來源、蒐集流程與使用限制，正好對應我在作業裡一直擔心的「資料合法性」問題。  
*採用方式*：我在期末專題的 data gating 層加入自動產出 datasheet/model card 的 pipeline，讓每次訓練都可重建資料血統。  
*參考文獻*：  
- [Gebru et al., 2018. "Datasheets for Datasets."](https://arxiv.org/abs/1803.09010)  
- [Mitchell et al., 2019. "Model Cards for Model Reporting."](https://dl.acm.org/doi/10.1145/3287560.3287596)

### 問題 2（Week 3-4：非 IID 數據下的梯度可信度）
*原始疑問*：當報價與風險訊號存在強烈的自相關與延遲，SGD 的估計還會保持一致嗎？  
*研究狀態*：已有研究。Lei、Wasserman 等人分析了在 mixing 條件下的 SGD 收斂性，並給出如何調整步長與小批次大小以應付相依樣本；Ovadia 等人則實際測試模型在 dataset shift 下的不確定性表現。  
*採用方式*：我參考這些結果，把小批次窗口固定在「資料 mixing 時間」之內，並且在每次 shift 被檢測到時重新估計學習率。  
*參考文獻*：  
- [Lei et al., 2020. "Stochastic Gradient Descent for Dependent Data."](https://arxiv.org/abs/2002.08537)  
- [Ovadia et al., 2019. "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift."](https://arxiv.org/abs/1906.02530)

### 問題 3（Week 5-6：把硬性監理約束納入訓練）
*原始疑問*：在推導 margin bound 與 kernelized feature map 後，我仍不知道要怎麼把「資本充足率」這種不可違反的硬性規則嵌入訓練。  
*研究狀態*：已有研究。Cotter 等人提出了對不可微分約束進行 Lagrangian 優化的方法；Achiam 等人的 Constrained Policy Optimization 則給出在 RL 場景中保持 constraint satisfaction 的做法，兩者都可轉成我期末專題的 constraint layer。  
*採用方式*：把 VaR、ESG、流動性門檻寫成懲罰項與投訴函式，把梯度拆成主問題（績效）與對偶問題（法遵），確保模型建議一定落在合法區域。  
*參考文獻*：  
- [Cotter et al., 2019. "Optimization with Non-Differentiable Constraints."](https://proceedings.mlr.press/v97/cotter19a.html)  
- [Achiam et al., 2017. "Constrained Policy Optimization."](https://arxiv.org/abs/1705.10528)

### 問題 4（Week 7-8：Diffusion/DSM loss 的時間權重怎麼選）
*原始疑問*：我在 score matching 作業裡還不確定每個時間步的權重要怎麼設，才能既維持穩定又兼顧尾端情境。  
*研究狀態*：已有研究。Nichol & Dhariwal 在 improved DDPM 中提出 cosine schedule 與 loss reweight；Karras 等人的 EDM 系統化分析了噪音排程、solver 與 weighting 的搭配方式，直接回答了「怎麼取樣才會穩」的疑問。  
*採用方式*：我把 EDM 的 sigma sampling 搭配 cosine-reweighted DSM loss，保留 Week 7 公式的嚴謹性，又能讓 stress-path 生成不會崩掉。  
*參考文獻*：  
- [Nichol & Dhariwal, 2021. "Improved Denoising Diffusion Probabilistic Models."](https://arxiv.org/abs/2102.09672)  
- [Karras et al., 2022. "Elucidating the Design Space of Diffusion-Based Generative Models."](https://arxiv.org/abs/2206.00364)

### 問題 5（Week 9-10：如何衡量 diffusion scenario 的覆蓋率與多樣性）
*原始疑問*：Week 10 的 probability-flow ODE 讓我可以生成 stress path，但還無法說服教授或監理「這些路徑真的覆蓋尾端」。  
*研究狀態*：已有研究。Sajjadi et al. 與 Kynkäänniemi et al. 提出的 Precision & Recall 以及 Improved PR 指標，可以用於衡量生成分布的 coverage 與 fidelity。我把 embedding 換成風險特徵（VIX、CDX、VaR），即可得到 tail coverage 報告。  
*採用方式*：將 diffusion 產生的情境投影到風險特徵空間，並計算 (Precision, Recall, F-score) 來佐證 scenario diversity。  
*參考文獻*：  
- [Sajjadi et al., 2018. "Assessing Generative Models via Precision and Recall."](https://arxiv.org/abs/1806.00035)  
- [Kynkäänniemi et al., 2019. "Improved Precision and Recall Metric for Assessing Generative Models."](https://openaccess.thecvf.com/content_ICCV_2019/html/Kynkaanniemi_Improved_Precision_and_Recall_Metric_for_Assessing_Generative_Models_ICCV_2019_paper.html)

### 問題 6（Open problem：LLM 協同監理敘事）
*原始疑問*：能否把 LLM 當成「監理聯絡窗口」，即時把每條 stress path 的違規原因翻成監理可讀的故事，並自動蒐證？  
*研究狀態*：目前尚無明確研究。雖然有零散的「LLM 當 compliance copilot」部落格文章，但我找不到經過同行評審或 arXiv 正式論文描述如何把 diffusion scenario、tamper-evident log 與監理問答全面串在一起。  
*口語化重述*：我們還沒有一套可證明可靠的做法，能讓聊天機器人讀懂複雜的壓力測試結果，然後自動寫出監理要的報告。眼下只能用人工覆核去補洞。  
*下一步*：把這個需求列成期末專題的研究延伸，尋找校內法遵或 FinTech 研究室是否願意與我們一起探索。

## 2. Toy model / Solvable model problem
延續 week10 期末專題的「跨資產風控協作中心」，Toy model 針對 8 個資產族群（VIX、CDX、亞洲 IG/HY、台股期權、國債、能源、碳權、外匯）及 4 類決策角色（交易、資金調度、合規、監理 liaison）建立 16 維輸入，並增加一個以開源 LLM 為核心的語義代理負責解析各 desk 的風控備忘。每筆資料包含即時報價、30 天滾動波動度、情緒分數、內部限額使用率，外加 ESG 違規旗標與監理詢問編號。流程分三層：data gating 以 SLAs、drift 指標與 LLM 產生的 provenance chain 判定可否進倉；model layer 將 score-based diffusion、tempered posterior 及 rule-based classifier 串成 mixture-of-samplers，產生 12,000 條跨資產 stress path，並由 consistency agent 自動撰寫交易敘事，再檢查 VIX<=35、CDX widen<=80 bps、單日 VaR<=資本 15% 等約束；deployment layer 將違規次數、指派調整、人工覆核與 tamper-evident log 上鏈，且即時推播給監理 liaison。評分指標除 coverage@tail、BCVaR delta、rule violation ratio 外，另加入 scenario overlap、narrative novelty、audit replay latency，確保樣本多樣且可在 5 分鐘內重播完整 trace，讓主管與監理單位抽查 sandbox。另設計分層式 RL 控制器模擬資金調度與保險庫互動，演練未來 24 小時流動性遷移並提供教授與同學檢視創新度。



所有回答都經過ChatgGPT和Gemini校稿潤飾
