# Autonomous-ML-Bidding-Agent-for-Used-Car-Auctions
An autonomous machine learning bidding agent for used car auctions. It utilizes a highly tuned LightGBM regression model to predict hammer prices and executes a deterministic, exponential decay bidding strategy to manage a $500,000 budget for maximum profit.

## Project Overview
This project involves the development of an autonomous bidding agent designed to participate in a live wholesale used car auction simulation. The primary objectives were to train a machine learning regression model to accurately predict the final selling price (hammer price) of a vehicle and to develop a deterministic bidding algorithm to manage a $500,000 budget to secure a profitable inventory.

## Data Cleaning & Feature Engineering
The data pipeline was strictly constrained to 11 basic features to ensure seamless interference and prevent data leakage.
* **Handling Missing Data:** Dropped rows where core identifiers (selling price, odometer, and year) were missing. Imputed remaining numerical features with the median or mode, and replaced missing categorical values with 'unknown'.
* **Outlier Handling:** Implemented a 1st and 99th percentile cutoff to trim anomalies while preserving the legitimate long-term luxury market, as used car prices are heavily right-skewed.
* **Dimensionality Reduction (80/20 Rule):** Retained only top high-frequency vehicle models and lumped the rest into an 'other' category to reduce noise and prevent the model from memorizing low-volume anomalies.
* **Encoding:** Applied a single `OrdinalEncoder` to all 8 categorical features simultaneously.

## Exploratory Data Analysis (EDA) Insights
* Vehicle pricing does not scale linearly with condition ratings. 
* There is a sharp, non-linear collapse in vehicle value when the condition score drops below 3.0.
* Cars rated 4.0 and above command a disproportionate premium, which was later integrated into the bidding agent's risk multiplier logic.

## Model Training & Performance
A gradient boosting framework using **LightGBM** was chosen for its superiority in handling high-cardinality tabular data and its fast execution speeds, which are critical for live automated bidding. The model was optimized using `RandomizedSearchCV` with a 3-fold cross-validation strategy on an 80/20 data split.

The hyperparameter tuning process drastically reduced error metrics, shrinking the average pricing error by over $400 per vehicle to provide a tighter, safer margin for the live bidding agent.

| Metric | Baseline Model | Tuned LightGBM Model | Improvement |
| :--- | :--- | :--- | :--- |
| **MAE** | $1794.11 | $1360.84 | -$433.27 |
| **RMSE** | $2632.29 | $2044.52 | -$587.77 |
| **R² Score** | 0.9039 | 0.9420 | +0.0381 |

## Deterministic Bidding Sequence
The `place_bid` function operates as a deterministic sequence driven by the agent's internal state to balance aggressive acquisition against bankroll protection. 
* **Profit Shield:** The agent operates on a hardcoded 13% baseline profit margin, which absorbs the model's $1360.84 MAE to ensure profitability.
* **Risk Multiplier:** When a vehicle's condition rating is $\ge 4.0$, a $1.05\times$ multiplier is applied to the predicted value, allowing the agent to bid 5% more on premium, low-risk inventory.

### Bidding Formula (Exponential Decay)
To close the gap aggressively in early rounds and intimidate simpler linear agents, the agent utilizes an exponential decay formula:

$$BID_{next} = BID_{current} + 100 + \left( \frac{DistanceToMax}{e^{k \cdot round}} \right)$$

*Where:*
*   $k = 0.6$
*   **DistanceToMax** = The remaining gap to the agent's absolute limit
*   **Round** = The current bid cycle

As auction rounds increase, the denominator grows exponentially, slowing down the bid increments drastically as the agent approaches its 13% margin limit. This prevents accidental overbidding caused by network latency or rapid counter-bids.

---

## Author
**Name:** Anjali Pogulwad  
**Program:** B.Tech in Data Science and Artificial Intelligence (DSAI)  
**Institution:** IIT Guwahati
