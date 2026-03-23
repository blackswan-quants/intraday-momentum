# Quantitative Analysis of Intraday Momentum via Volatility Regimes, Trend Filtering, and Temporal Persistence

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![QuantConnect](https://img.shields.io/badge/QuantConnect-Lean-orange.svg)

[📄 Read the Research Paper Here](#)

**Abstract:** We present an empirical examination of five intraday momentum trading iterations (Strategies 0–4), evaluating the impact of discrete execution frequencies, moving average filters, sequential persistence validations, and their combination on risk-adjusted performance. The underlying alpha thesis capitalises on short-term trend extensions measured against a dynamically calculated intraday volatility threshold ($\sigma$). Progressing from a symmetric baseline execution protocol to an asymmetric, temporally persistent framework with entry-quality filtering, the study isolates the **Momentum Paradox**: aggressive risk management without a noise filter destroys the very returns it seeks to protect, while the composition of an entry pre-filter and an exit post-filter produces the Pareto-dominant outcome.

---

## 1. Introduction and Core Methodology

The primary objective of intraday momentum strategies is to identify and participate in structural market movements, extracting excess returns during periods of directional dislocation. However, high-frequency price data is innately noisy, characterised by continuous mean-reverting microstructure fluctuations.

The baseline logic relies on volatility-normalised breakout boundaries. At each evaluation interval, statistical upper and lower thresholds are derived from the 14-day rolling mean of minute-by-minute price deviations from the daily open. The strategy employs a dynamic position sizing mechanism, inversely scaling leverage to target a constant daily portfolio volatility of 2.0%:

$$\Lambda = \min\!\left(\text{Max}_L,\; \frac{\sigma_{\text{target}}}{\sigma_{\text{daily}}}\right)$$

optimising the risk-budgeting allocation across changing volatility regimes. All positions are compulsorily liquidated at 15:58 ET to eliminate overnight gap risk.

We trace the evolution of this core alpha engine through four successive modifications that progressively introduce structural controls governing entry and risk-management decision criteria.

---

## 2. Empirical Model Evaluation

### 2.1 Strategy 0: The Symmetric Baseline Model

**Framework:** Strategy 0 (`strategy0.py`) employs a symmetric observation cycle. The algorithm evaluates both entry and exit conditions on a uniform 30-minute interval ($\Delta t_{\text{exec}} = 30\,\text{min}$). Positions are initiated upon breaking dynamic historical volatility boundaries and liquidated when price action reverts across the VWAP or crosses the opposite boundary.

**Financial Rationale:** This establishes a solid foundational alpha, capitalising on established half-hour momentum segments. The symmetric 30-minute resolution, however, creates structural lag during intraday reversals: once a trend reversal is underway, the strategy can sustain up to 30 minutes of adverse price action before the exit condition is re-evaluated. The strategy additionally enters breakout trades irrespective of the broader intraday trend structure, generating false positives at maximum rate during mean-reverting regimes.

---

### 2.2 Strategy 1: Decoupled Frequencies and Granular State Observation

**Framework:** Strategy 1 (`strategy1.py`) decouples the decision frequencies. Entries remain constrained to 30-minute intervals, but the exit monitoring cycle is increased sixfold to a 5-minute resolution ($\Delta t_{\text{exit}} = 5\,\text{min}$), reducing the maximum adverse excursion window from 30 to 5 minutes.

**Financial Rationale:** Increasing the resolution of the risk-management layer contains single-trade tail risk, reducing maximum drawdown from 11.2% to 10.1%. However, over-sampling a noisy time series without a filter triggers heavy whipsawing: VWAP generates brief, transient violations with no predictive value for the ultimate direction of the trade. Strategy 1 systematically truncates valid winners on these transient violations, collapsing the win rate from 40% to 32% and reducing total return from 196% to 177%. The lesson: monitoring speed alone is insufficient without a filter for statistical significance.

---

### 2.3 Strategy 2: Dual-Confirmation Regime Filtering

**Framework:** Strategy 2 (`strategy2.py`) reverts to a symmetric 30-minute cadence for both entries and exits, and introduces a 100-period Exponential Moving Average (EMA) on 1-minute bars as a structural trend-confirmation filter. Positions require dual confirmation: breaking the statistical $\sigma$-boundary **and** directional alignment with the EMA:

$$\text{Long entry:}\quad P_t > \text{UB}_t \;\text{ and }\; P_t > \text{EMA}_{100}(t)$$
$$\text{Short entry:}\quad P_t < \text{LB}_t \;\text{ and }\; P_t < \text{EMA}_{100}(t)$$

**Financial Rationale:** The EMA acts as a structural low-pass boundary, screening out short-term, low-probability mean-reverting deviations. The 100-minute lookback bridges the gap between microstructure noise (resolved in the first 20–30 minutes) and the structural intraday trend (typically persisting 2–4 hours). This entry filter reduces total orders from ~3,600 to ~3,450, recovers the win rate to 40%, and yields the best maximum drawdown of any single-mechanism strategy (8.7%), with total return rising to 227% and Sharpe to 0.94. Its residual limitation is slow exit cadence: positions remain exposed for up to 30 minutes after a reversal begins.

---

### 2.4 Strategy 3: Algorithmic Insulation via Temporal Persistence

**Framework:** Strategy 3 (`strategy3.py`) reinstates the decoupled 5-minute exit loop from Strategy 1, but embeds an $N=4$ **Exit Persistence Protocol**. A composite exit condition violation (VWAP or band crossing) does not trigger immediate liquidation; the violation must persist sequentially across four consecutive 5-minute checks (20 continuous minutes) before the position is closed:

$$\text{Exit triggered} \iff \sum_{k=0}^{3} \mathbf{1}\!\left[s(\tau - k \cdot 5\,\text{min}) = 1\right] = 4$$

**Financial Rationale:** The 4-bar sequential validation functions as a causal low-pass filter on the binary exit signal, with an effective cutoff at periods shorter than 20 minutes. Noise components with periods below this threshold are attenuated; genuine structural reversals persisting ≥20 minutes pass through unimpeded. This configuration yields the highest total return among single-mechanism strategies (252%), crossing the institutional Sharpe threshold of 1.0 (Sharpe 1.012), while holding drawdown at 9.3%. Its remaining limitation is the absence of an entry-quality filter: it benefits from a noise-robust exit mechanism but remains exposed to directionally misaligned entries during mean-reverting regimes.

---

### 2.5 Strategy 4: Combined EMA Filter and Temporal Persistence

**Framework:** Strategy 4 (`strategy4.py`) activates both mechanisms simultaneously, synthesising the orthogonal improvements of Strategies 2 and 3. Entry requires a volatility breakout **and** EMA confirmation (as in Strategy 2). Exit monitoring runs at the 5-minute cadence with the 4-bar persistence gate (as in Strategy 3):

$$\text{Long entry:}\quad P_t > \text{UB}_t \;\text{ and }\; P_t > \text{EMA}_{100}(t)$$
$$\text{Exit triggered} \iff \sum_{k=0}^{3} \mathbf{1}\!\left[s(\tau - k \cdot 5\,\text{min}) = 1\right] = 4$$

**Financial Rationale:** The two mechanisms are structurally independent and address orthogonal failure modes: the EMA filter is a **pre-filter** that selects only breakout signals confirmed by the slower-timescale trend, while the persistence gate is a **post-filter** that suppresses transient exit-signal noise below the 20-minute threshold. Their combination is synergistic rather than merely additive — higher-quality entries are held longer against noise, and the persistence gate prevents premature liquidation of the premium trades that pass the EMA screen. The result is the Pareto-dominant outcome across all five versions: total return 259%, Sharpe 1.036, Sortino 1.438, maximum drawdown 8.4%, and a Probabilistic Sharpe Ratio of 83.4% (clearing the 80% institutional confidence threshold). Expectancy of 0.220 per trade exceeds every predecessor, driven by a higher average win (0.74% vs. 0.68% baseline) reflecting the selection effect of dual-confirmation entry.

---

## 3. Concluding Analytics

The empirical progression outlines the defining dilemma of algorithmic momentum implementation: distinguishing structural dislocation from stationary noise. Strategy 1 demonstrates that speed without a noise filter is destructive. Strategy 2 demonstrates that entry quality compounds multiplicatively. Strategy 3 demonstrates that exit robustness via temporal persistence unlocks the Sharpe 1.0 threshold. Strategy 4 closes the loop by combining both insights, confirming that pre-filtering entries and post-filtering exit signals are structurally independent improvements whose composition is strictly superior to either mechanism in isolation.

---

### Annex A: 8-Year Backtest Summary Performance Matrix (May 2017 – May 2025)

| Strategy | Total Return | Sharpe | Max Drawdown | Win Rate | Total Orders |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy 0** | 196.21% | 0.835 | 11.20% | 40% | 3,594 |
| **Strategy 1** | 177.00% | 0.84 | 10.10% | 32% | 4,900 |
| **Strategy 2** | 227.00% | 0.94 | 8.70% | 40% | 3,450 |
| **Strategy 3** | 252.00% | 1.012 | 9.30% | 40% | 3,550 |
| **Strategy 4** | **259.24%** | **1.036** | **8.40%** | **40%** | **3,388** |

### Annex B: 3-Month Backtest Summary Performance Matrix (Feb – Apr 2025)

| Strategy | Total Return | Sharpe | Max Drawdown | Win Rate | Total Orders |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Strategy 0** | 4.70% | 0.826 | 5.40% | 39% | 122 |
| **Strategy 1** | 4.79% | 0.91 | 5.50% | 28% | 158 |
| **Strategy 2** | 5.79% | 1.14 | 5.10% | 40% | 116 |
| **Strategy 3** | 6.36% | 1.30 | 5.10% | 41% | 116 |
| **Strategy 4** | **6.30%** | **1.29** | **5.10%** | **40%** | **114** |