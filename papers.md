# Combining deep reinforcement learning with technical analysis and trend monitoring on cryptocurrency markets

> **Note:** This is a summary of the original paper. In this PyTorch implementation, the **Smurf mechanism is NOT implemented** as the position-based trading environment with SL/TP, drawdown penalties, and the FLAT action already provides sufficient risk management.

**Authors:** Vasileios Kochliaridis, Eleftherios Kouloumpris, Ioannis Vlahavas  
**Journal:** Neural Computing and Applications (2023) 35:21445–21462  
**Received:** 30 December 2022 / **Accepted:** 21 March 2023 / **Published online:** 20 April 2023  
**DOI:** https://doi.org/10.1007/s00521-023-08516-x

---

## Abstract

Cryptocurrency markets experienced a significant increase in popularity, which motivated many financial traders to seek high profits in cryptocurrency trading. The predominant tool that traders use to identify profitable opportunities is technical analysis. Some investors and researchers also combined technical analysis with machine learning, in order to forecast upcoming trends in the market. However, even with the use of these methods, developing successful trading strategies is still regarded as an extremely challenging task. Recently, deep reinforcement learning (DRL) algorithms demonstrated satisfying performance in solving complicated problems, including the formulation of profitable trading strategies. While some DRL techniques have been successful in increasing profit and loss (PNL) measures, these techniques are not much risk-aware and present difficulty in maximizing PNL and lowering trading risks simultaneously. 

This research proposes the combination of DRL approaches with rule-based safety mechanisms to both maximize PNL returns and minimize trading risk. First, a DRL agent is trained to maximize PNL returns, using a novel reward function. Then, during the exploitation phase, a rule-based mechanism is deployed to prevent uncertain actions from being executed. Finally, another novel safety mechanism is proposed, which considers the actions of a more conservatively trained agent, in order to identify high-risk trading periods and avoid trading. Our experiments on 5 popular cryptocurrencies show that the integration of these three methods achieves very promising results.

**Keywords:** Deep reinforcement learning, Machine learning, Risk optimization, Distributional DQN, Soft Actor-Critic, Trading, Technical analysis.

---

## 1. Introduction

Cryptocurrencies are digital currencies that circulate through a computer network, which is not reliant on any central authority [1]. Over the last few years, both popularity and value of this type of technology has risen, so many traders and investors have shifted their attention to trading cryptocurrency assets, such as Bitcoin. Cryptocurrency assets are traded in a similar manner to how stocks are traded, but are fundamentally different [2]. First of all, cryptocurrency markets are accessible every hour. Secondly, there are no intermediaries involved in cryptocurrency transactions, so the transaction costs could be lower. Finally, cryptocurrency markets are characterized by their tremendous volatility and rapid fluctuations. For all these reasons, cryptocurrency markets provide traders with great money-earning opportunities, but also involve higher risk [2].

Nowadays, financial markets information spreads easier and more quickly than ever before. As a result, numerous professional investors and traders use technical analysis, which is a tool that is applied on past market data and allows traders to forecast market trends. Technical analysis provides technical indicators, which are pattern-based indications of an asset’s momentum, volatility and trend [3]. However, technical indicators are prone to producing false trend signals, so investors and financial analysts usually combine a set of technical indicators [3].

[...] In this paper, we extend the methodology of **TraderNet-CR** [8], which is a trading system composed of three modules:
1.  **TraderNet:** A DRL agent trained using a novel reward function named *Round-Trip Strategy*. In this work, we modified it to combine both market and limit orders.
2.  **N-Consecutive:** A rule-based safety mechanism which inspects a window of TraderNet's previous actions to prevent uncertain actions.
3.  **Smurfing:** A mechanism using a secondary agent ("Smurf") trained more conservatively to detect high-risk periods.

---

## 2. Related Work

* **Huang et al. [9]:** Investigated cryptocurrency return predictability using a tree-based model trained on 124 technical indicators.
* **Guarino et al. [4]:** Compared algorithmic trading agents using technical analysis with DRL agents. Found that DRL agents used indicators more efficiently but lacked explainability.
* **Satarov et al. [6]:** Applied DQN to identify profitable trading points. Showed RL performed better than traditional strategies given low fees (0.15%).
* **Mahayana et al. [10]:** Applied a clipped policy-gradient agent for BTC trading using 1-minute candlesticks. Failed to outperform buy and hold strategy when transaction costs were considered.
* **Schnaubelt [7]:** Applied a clipped policy-gradient method on limit orders, reducing transaction costs by up to 36.93%.
* **Li et al. [11]:** Proposed a transformer-based architecture (LSRE-CAAN) for high-frequency trading.
* **Lucarelli and Borrotti [12]:** Employed a multi-agent framework for portfolio management.
* **Cui et al. [13]:** Used a CVaR-aware policy-gradient method.

**Gap Analysis:** Previous works often disregarded transaction costs, prioritized profit over risk minimization, or lacked safety mechanisms. Our work aims to improve by combining market data with technical/social indicators, using high-performance DRL algorithms (QR-DQN and Categorical SAC) with a novel risk-adjusted reward function, and adding safety mechanisms.

---

## 3. Background

### 3.1 Technical Analysis

We define a subset of popular technical indicators used in this work:

**Definition 1 (Exponential Moving Average - EMA):**
$$EMA(n) = Price(n) \cdot k + EMA(n-1) \cdot (1-k)$$
where $k = \frac{2}{N+1}$.

**Definition 2 (Double Exponential Moving Average - DEMA):**
$$DEMA_N = 2EMA_N - EMA \text{ of } EMA_N$$

**Definition 3 (MACD):**
$$MACD = EMA_{12} - EMA_{26}$$

**Definition 4 (Aroon):**
$$Aroon_{UP} = \frac{25 - \text{Period since new High}}{25} \times 100$$
$$Aroon_{DOWN} = \frac{25 - \text{Period since new Low}}{25} \times 100$$

**Definition 5 (Commodity Channel Index - CCI):**
$$CCI = \frac{TP(n) - 20\text{-Period } EMA_{TP}}{0.015 \times \text{Mean Deviation}}$$
where $TP(n) = \frac{High(n) + Low(n) + Close(n)}{3}$.

**Definition 6 (Average Directional Index - ADX):**
$$ADX = MA \times \frac{PDI - NDI}{PDI + NDI} \times 100$$

**Definition 7 (Stochastic Oscillator - STOCH):**
$$\frac{Close(n) - L_{14}}{H_{14} - L_{14}}$$
where $L_{14}$ and $H_{14}$ are the lowest low and highest high of the last 14 periods.

**Definition 8 (Relative Strength Index - RSI):**
$$RSI = 100 - \frac{100}{1 + RS}$$
where $RS = \frac{\text{Average Gain}}{\text{Average Loss}}$.

**Definition 9 (On-Balance Volume - OBV):**
Uses volume flow to predict price changes based on whether the close price increased or decreased compared to the previous period.

**Definition 10 (Bollinger Bands):**
$$BBAND_{UP} = Mean(TP) + 2 \cdot Std(TP)$$
$$BBAND_{DOWN} = Mean(TP) - 2 \cdot Std(TP)$$

**Definition 11 (Volume-Weighted Average Price - VWAP):**
$$VWAP(n) = \frac{\sum (Close(i) \cdot Volume(i))}{\sum Volume(i)}$$

**Definition 12 (Accumulation/Distribution Line - ADL):**
$$ADL(n) = ADL(n-1) + \frac{(Close - Low) - (High - Close)}{High - Low} \cdot Volume$$

### 3.2 Reinforcement Learning (RL)

The problem is formulated as a Markov Decision Process (MDP). The goal is to maximize the expected discounted cumulative return:
$$V_{\pi}(s) = E[\sum_{t=0}^{T}\gamma^{t}r_{t+1}|s_{0}=s]$$

#### 3.2.1 Distributional RL (QR-DQN) and Categorical SAC
- **QR-DQN:** models the return distribution via quantile regression Huber loss:
  $$L = \frac{1}{N} \sum_{i,j} |\tau_j - \mathbb{1}_{\delta_{ij}<0}| \cdot \text{Huber}_\kappa(\delta_{ij})$$
  where $\delta_{ij}$ is the TD error between target and predicted quantiles.
- **Categorical SAC:** maximizes expected return plus entropy with twin Q-networks and a categorical policy; temperature $\alpha$ is learned to match an entropy target, while target networks are updated softly with rate $\tau$.

### 3.3 TraderNet-CR

**Problem Formulation:**
* **Action Space:** $A = \{BUY, SELL, HOLD\}$
* **State Space:** $S \in R^{N \times 21}$. A sequence of ordered vectors from $N$ previous 1-hour intervals. Each vector has 21 features (market data, technical indicators, and Google Trends score).

**Original Round-Trip Strategy:**
The agent is rewarded immediately after opening a round-trip based on the maximum possible return within the next $K$ hours.

---

## 4. Methodology

### 4.1 Modified Round-Trip Strategy

We modified the strategy to use **Logarithmic PNL returns** and combine **Market orders** (entry) with **Limit orders** (exit).

The reward function is:

$$
r_{t+1} = \begin{cases} 
\ln\left(\frac{H_{t_{max}} - f \cdot H_{t_{max}}}{C_t + f \cdot C_t}\right) & \text{if } a_t = \text{BUY} \\
\ln\left(\frac{C_t + f \cdot C_t}{L_{t_{min}} - f \cdot L_{t_{min}}}\right) & \text{if } a_t = \text{SELL} \\
-\max(r_{t(a)}) & \text{if } a_t = \text{HOLD} 
\end{cases}
$$

Using the property $\ln(A \cdot B) = \ln A + \ln B$, we can sum these logarithmic returns to get the total cumulative return. The use of log returns ensures "raw-log equality" for small returns ($\log(1+r) \approx r$), motivating the agent to avoid uncertain trading periods where potential profit is small.

### 4.2 Smurfing

A secondary agent ("Smurf") is trained to detect high-risk states.

* **Higher Fees:** Trained with fees $f' > f$.
* **Positive Hold Reward:** Reward for holding is a small positive constant $w \ll 1$.

**Smurf's Reward:**

$$
r'_{t+1} = \begin{cases} 
\ln\left(\frac{H_{t_{max}} - f' \cdot H_{t_{max}}}{C_t + f' \cdot C_t}\right) & \text{if } a_t = \text{BUY} \\
\ln\left(\frac{C_t + f' \cdot C_t}{L_{t_{min}} - f' \cdot L_{t_{min}}}\right) & \text{if } a_t = \text{SELL} \\
w & \text{if } a_t = \text{HOLD} 
\end{cases}
$$

### 4.3 The Integrated TraderNet-CR

In this paper, we integrate all the aforementioned modules into a single integrated agent.

1.  **Smurf Check:** During the exploitation phase, we use Smurf's policy to determine whether opening a round-trip at a state $s_t$ has high PNL potential. If Smurf selects HOLD, the system holds.
2.  **TraderNet Decision:** If Smurf allows trading, we use TraderNet's policy to select an action $a_t$ (BUY or SELL).
3.  **N-Consecutive Rule:** Finally, we use the N-Consecutive mechanism. The action $a_t$ is executed only if the agent suggested the same action for the previous $N$ consecutive steps ($a_t = a_{t-1} = \dots = a_{t-N+1}$).

---

## 5. Experiments and Results

### 5.1 Datasets
* **Assets:** BTC, ETH, ADA, LTC, XRP.
* **Source:** CoinAPI (OHLCV) and Google Trends (Social indicator).
* **Timeframe:** 2016 to Nov 2022.
* **Preprocessing:** Min-Max scaling to range [0, 1.0]. Sequence length $N=12$, Horizon $K=20$.

### 5.2 Experiment Setup
* **Training/Evaluation:** Evaluation on the final 2250 hours (approx 3 months). Different evaluation timelines were used for each asset to avoid overlapping market conditions.
* **Fees:** 1.0% commission fees simulated.
* **Metrics:** Cumulative Returns (CR), Cumulative PNL (CP), Investment Risk (IR), Sharpe Ratio (SHR), Sortino Ratio (SOR), Maximum Drawdown (MDD).

### 5.3 Hyper-parameter Tuning

**Table 1: QR-DQN Hyper-parameters**

| Parameter | Value |
| :--- | :--- |
| Num quantiles | 51 |
| Mini-batch size | 128 |
| Discount factor $\gamma$ | 0.99 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Target update interval | 2000 |
| Huber $\kappa$ | 1.0 |
| Prioritized replay $\alpha$ | 0.6 |
| Prioritized replay $\beta$ start | 0.4 |

**Table 2: Double DQN Hyper-parameters**

| Parameter | Value |
| :--- | :--- |
| Exploration $\epsilon$ | 0.1 |
| Batch size | 64 |
| Target network update | 3000 |
| Discount factor $\gamma$ | 0.99 |
| Q-network layers | Same as Actor |

### 5.4 Modified Round-Trip Strategy Evaluation

**Table 3: Modified Round-Trip Strategy Performance** (Selected Data)

| Market | Strategy | CR | CP | IR | SHR | SOR | MDD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BTC** | Standard | 1.186 | 1.186 | 0.617 | 1.030 | 1.097 | 0.978 |
| | **Modified** | **13.173** | **13.173** | **0.477** | **1.357** | **4.475** | **0.357** |
| **ETH** | Standard | 17.237 | 17.249 | 0.472 | 1.365 | 3.035 | 0.581 |
| | **Modified** | **35.808** | **35.808** | **0.331** | **1.776** | **54.248** | **0.112** |

*Note: The Modified Round-Trip Strategy outperformed the previous strategy in all market environments, achieving significantly higher PNL and lower risk.*

### 5.5 N-Consecutive Evaluation

Experiments with $N=\{1, 2, 3, 4, 5\}$.
* Larger window sizes generally resulted in lower Investment Risk (IR) and MDD.
* Window sizes between 2 and 3 achieved a satisfying balance between risk and profit.
* Too large windows (e.g., 5) caused missed profit opportunities (lower PNL).

### 5.6 Smurfing Evaluation

**Table 5: Smurfing Performance** (Comparison)

| Market | Module | CR | MDD |
| :--- | :--- | :--- | :--- |
| **BTC** | TraderNet | 13.173 | 0.357 |
| | TraderNet + Smurf | 22.734 | **0.289** |
| **ETH** | TraderNet | 35.808 | 0.112 |
| | TraderNet + Smurf | 28.370 | **0.035** |

*Smurfing effectively reduced IR and MDD metrics on all markets.*

### 5.7 Integrated TraderNet-CR Evaluation

The Integrated TraderNet-CR (combining QR-DQN/Categorical SAC heads, Modified Round-Trip, Smurfing, and N-Consecutive) was compared against TraderNet-Smurf and a DDQN version.
* The integrated approach results in lower but steadily increasing profits over the trading period.
* It avoids high-risk trading and makes more certain actions.
* DDQN showed sub-optimal performance compared to the distributional and entropy-regularized variants in most experiments.

---

## 6. Conclusion and Future Work

We demonstrated that integrating several modules into TraderNet-CR achieves high PNL performance while reducing trading risk.
1.  **Modified Round-Trip Strategy** significantly increased returns.
2.  **N-Consecutive** mechanism reduced uncertainty.
3.  **Smurfing** helped identify high-risk periods.

Future work could involve adding more technical indicators, experimenting with Apex-DQN or Soft Actor-Critic (SAC), and performing feature importance analysis to simplify the state space.

---

## References

1.  Nakamoto S (2008) Bitcoin: a peer-to-peer electronic cash system. Decent Bus Rev 21260.
2.  Fang F et al (2022) Cryptocurrency trading: a comprehensive survey. Financ Innov 8(1):1–59.
3.  Lin TC (2012) The new investor. UCLA L Rev 60:678.
4.  Guarino A et al (2022) To learn or not to learn? Evaluating autonomous, adaptive, automated traders... Neural Comput Appl.
5.  Arratia A, López-Barrantes AX (2021) Do google trends forecast bitcoins? J Bank Financ Technol.
6.  Sattarov O et al (2020) Recommending cryptocurrency trading points with deep reinforcement learning approach. Appl Sci.
7.  Schnaubelt M (2022) Deep reinforcement learning for the optimal placement of cryptocurrency limit orders. Eur J Oper Res.
8.  Kochliaridis V, Kouloumpris E, Vlahavas I (2022) Tradernet-cr: cryptocurrency trading with deep reinforcement learning.
9.  Huang J-Z et al (2019) Predicting bitcoin returns using high-dimensional technical indicators. J Finance Data Sci.
10. Mahayana D et al (2022) Deep reinforcement learning to automate cryptocurrency trading. IEEE.
11. Li J et al (2023) Online portfolio management via deep reinforcement learning with high-frequency data. Inf Process Manag.
12. Lucarelli G, Borrotti M (2020) A deep Q-learning portfolio management framework for the cryptocurrency market. Neural Comput Appl.
13. Cui T et al (2023) Portfolio constructions in cryptocurrency market: a CVaR-based deep reinforcement learning approach. Econ Model.
14. Pring MJ (1991) Technical analysis explained. McGraw-Hill.
15. Sutton RS, Barto AG (2018) Reinforcement learning: an introduction. MIT Press.
16. Lazaridis A et al (2020) Deep reinforcement learning: a state-of-the-art walkthrough. J Artif Intell Res.
17. Schulman J et al (2017) Proximal policy optimization algorithms. arXiv preprint.
