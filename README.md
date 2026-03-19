# Demand Forecasting with Multi-Factor Analysis

A machine learning framework that **quantifies the relative influence of five key factor categories on consumer demand** — going beyond prediction to answer *which factors matter most, and by how much*.

---

## Overview

Most demand forecasting systems tell you *what* demand will be. This project answers *why* — by systematically measuring the contribution of weather, economic conditions, social media trends, financial markets, and calendar effects to demand patterns using a multi-model interpretable ML approach.

Built as an academic study project at BITS Pilani (Mathematics Dept.), this framework is designed to be industry-adaptable and business-interpretable.

---

## Problem Statement

Businesses monitor many demand signals — weather, sentiment, macro indicators — but lack a **quantitative basis for prioritizing** which factors deserve investment. This project answers:

- How much does each factor category influence demand?
- Which signals should forecasting models prioritize?
- How can ML provide *interpretable, quantified* insights — not just predictions?

---

## Repository Structure

```
Demand-Forecasting/
│
├── m3sop_final_code.ipynb      # Full pipeline: data → features → models → analysis
│
├── reports/
│   └── report.pdf              # Full project report
│
├── presentation/
│   └── presentation.pdf        # Project presentation slides
│
├── LICENSE                     # All rights reserved
└── README.md
```

---

## Methodology

### Data Architecture

Five factor categories integrated from real APIs (with synthetic fallback):

| Factor Group | Features | Source |
|---|---|---|
| **Weather** | Temperature, Humidity, Pressure, Feels-like, Heat Index, MAs, Lags | OpenWeather API / Synthetic |
| **Economic** | GDP Growth, Inflation, Unemployment, Interest Rate, Consumer Confidence | Synthetic (India-calibrated) |
| **Social Media** | Shopping/Smartphone/Entertainment Trends, Sentiment, Engagement, Interactions | Synthetic |
| **Financial** | Stock Price, Volume, High/Low/Open, Volatility, Price Change, MAs, Lags | Alpha Vantage API / Synthetic |
| **Calendar** | Holiday flag, National/Religious indicators, Weekend, Month, Day-of-week, Quarter | Python `holidays` library |

> Pipeline auto-falls back to calibrated synthetic generation if API keys are unavailable.

### Feature Engineering

47 features engineered across 6 groups, including:
- 7-day and 30-day rolling averages for weather, financial, and social signals
- Lag features (1-day, 7-day) for time-series awareness
- Interaction features: heat index, sentiment × shopping trend, GDP × interest rate
- Full temporal decomposition: quarter, week-of-year, day-of-year

All features standardized via `StandardScaler` (RF/SHAP) or `MinMaxScaler` (LSTM).

### Demand Generation

Synthetic demand uses **controlled, known coefficients** enabling ground-truth validation:

```python
base_demand         = 1,500 units
GDP Growth          = +40 units/point
Inflation           = −25 units/point
Unemployment        = −20 units/point
Consumer Confidence = +4 units/point above 50
Shopping Trend      = +6 units/trend point
Smartphone Trend    = +4 units/trend point
National Holiday    = +500 units
Other Holiday       = +250 units
Weekend (Sat/Sun)   = +200 / +150 units
Festival Months     = +300 units (Sep–Nov)
Noise               = ±75 units (Gaussian, seed=42)
```

### Models

| Model | Configuration | Purpose |
|---|---|---|
| **Random Forest** | 150 estimators, max_depth=12, chronological 85/15 split | Primary feature importance |
| **SHAP on RF** | `TreeExplainer` on trained RF | Model-agnostic interpretability |
| **LSTM** | 2-layer (150→75 units), Dropout=0.3, 30-step sequences, EarlyStopping | Temporal sequence modelling |

---

## Results

### Model Performance

| Metric | Random Forest | LSTM |
|---|---|---|
| R² Score | 0.5029 | **0.5419** |
| MAE | 119.76 | **119.50** |
| RMSE | 153.07 | **146.94** |

LSTM outperforms Random Forest across all metrics, capturing temporal dependencies that tree-based models miss. Both models provide a solid baseline for multi-factor demand modelling on synthetic data.

### Factor Group Importance

Factor importance is aggregated by summing feature-level scores within each group. Measured using both RF default importance and SHAP mean absolute values:

**Consistent finding across both methods:** Social media trends and temporal patterns are the dominant demand drivers, followed by weather and financial signals — challenging traditional macro-economic-centric forecasting assumptions.

> Full pie chart breakdowns with precise percentages are available in the notebook outputs and project report.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML / Deep Learning | scikit-learn, TensorFlow / Keras |
| Interpretability | SHAP (`TreeExplainer`) |
| Data Processing | pandas, NumPy |
| APIs & Data | OpenWeather API, Alpha Vantage API, Python `holidays` |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Haryaksh1/Demand-Forecasting.git
cd Demand-Forecasting

# Install dependencies
pip install pandas numpy scikit-learn shap tensorflow matplotlib seaborn holidays requests

# Add your API keys in the notebook (optional — synthetic fallback works without them)
# OPENWEATHER_API_KEY = "your_key_here"
# ALPHAVANTAGE_API_KEY = "your_key_here"

# Run
jupyter notebook m3sop_final_code.ipynb
```

---

## Limitations

- Synthetic data may not capture all real-world demand complexities
- Factor importance assumes temporal stationarity
- Calibrated to Indian market conditions (Delhi weather base, India public holidays)
- SHAP for LSTM skipped due to TensorFlow/Keras version compatibility
- 730-day observation window

---

## Future Work

- Real-world validation with actual retail or e-commerce data
- Time-varying (rolling window) factor importance
- Causal inference beyond correlation
- LSTM SHAP integration with compatible explainer
- Industry-specific calibration

---

## License

**All Rights Reserved.**

This repository is shared for **viewing and academic reference only.**
No part of this code, methodology, or documentation may be copied, modified, redistributed, or used in any form — commercial or otherwise — without explicit written permission from the author.

© 2025 Haryaksh Manuh Bhardwaj. All rights reserved.

---

## References

1. Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
2. Ke et al. (2017) — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
3. Chen et al. (2025) — *Supply Chain Demand Forecasting based on Multi-Time Scale Data Fusion.* Computers & Industrial Engineering.
4. Tadayonrad & Ndiaye (2023) — *A new KPI model for demand forecasting in inventory management.* Supply Chain Analytics.
5. Zheng & Casari (2018) — *Feature Engineering for Machine Learning.* O'Reilly Media.
6. Choi & Varian (2012) — *Predicting the Present with Google Trends.* Economic Record.
7. MIT Sloan — *What is synthetic data and how can it help you competitively?*
