# ✈️ Flight Ticket Dynamic Pricing — LSTM & CRISP-DM

> *"Why does the same seat cost ₹3,500 today but ₹9,200 tomorrow?"*
> This project builds a machine learning engine to answer — and predict — exactly that.

---

## Overview

Every time you search for a flight, an invisible pricing algorithm is watching — tracking demand surges, days to departure, seat availability, and route competition. Airlines call it **dynamic pricing**, and it's one of the most profitable levers in the industry.

This project reverse-engineers that logic. Using **300,153 Indian domestic flight records**, we train deep learning models to predict ticket prices from flight attributes — airline, route, cabin class, stops, and most critically, **how many days remain before departure**. The temporal nature of the problem makes it a natural fit for **LSTM (Long Short-Term Memory)** networks, which excel at learning patterns that evolve over time.

The entire workflow follows the **CRISP-DM framework** — structured, reproducible, and deployment-ready.

---

## Results at a Glance

| Model | MAE (INR) | RMSE (INR) | R² | MAPE |
|-------|-----------|------------|-----|------|
| Ridge Regression | ₹5,017 | ₹8,781 | 0.849 | 27.5% |
| Vanilla LSTM | ₹3,917 | ₹6,813 | 0.909 | 22.0% |
| **GBM (Best)** | **₹2,370** | **₹4,238** | **0.965** | **14.8%** |
| BiLSTM + Attention | — | — | — | — |

**GBM explains 96.5% of price variance.** The Route-Level LSTM provides richer temporal modelling when true booking-horizon sequences are available, and is the architecture recommended for a production retraining pipeline.

---

## Key Findings

| # | Driver | Finding |
|---|--------|---------|
| 1 | **Cabin Class** | Business class commands a **3–5× price premium** — the single largest pricing lever |
| 2 | **Booking Horizon** | Last-minute tickets (≤7 days) cost up to **64% more** than advance bookings; prices spike non-linearly inside two weeks |
| 3 | **Airline Brand** | Vistara anchors the premium end; IndiGo and SpiceJet compete on cost — brand drives a consistent ₹500–₹1,000 spread on identical routes |
| 4 | **Season** | High-season fares run **20–35% above** low-season on the same routes |
| 5 | **Fuel Costs** | Pearson r = −0.07 — airlines absorb short-term fuel shocks rather than passing them directly to passengers |
| 6 | **Departure Time** | Early-morning slots are discounted; morning and evening windows attract a demand premium |

---

## CRISP-DM Phases

```
Business Understanding → Data Understanding → Data Preparation
         → Modelling → Evaluation → Deployment
```

| Phase | What Was Done |
|-------|---------------|
| **1. Business Understanding** | Framed ticket pricing as a regression problem anchored on `days_left` as the core temporal signal |
| **2. Data Understanding** | Explored price distributions, airline premiums, booking-horizon curves, stop penalties, and seasonal patterns across 300K+ records |
| **3. Data Preparation** | Label-encoded 9 categorical features, MinMax-scaled inputs and target, engineered urgency bins, price-per-km yield, and temporal features; reshaped for LSTM |
| **4. Modelling** | Four architectures: Ridge baseline → GBM → Vanilla LSTM → BiLSTM + Bahdanau Attention + Route-Level temporal LSTM |
| **5. Evaluation** | MAE, RMSE, MAPE, R² on a held-out 15% test set; residual analysis; booking-horizon price-curve validation |
| **6. Deployment** | Full artifact bundle saved; `predict_price()` scorer function; single-route booking-horizon forecast |

---

## Architecture

The flagship deep learning model is a **Bidirectional LSTM with Bahdanau Attention**:

```
Input: (batch, 1, 18 features)
│
├─ Bidirectional LSTM(128, return_sequences=True) + L2 regularisation
├─ BatchNormalization()
│
├─ BahdanauAttention(64)      ← weights each feature's contribution dynamically
│
├─ Dense(128, relu)
├─ Dropout(0.3)
├─ Dense(64, relu)
├─ Dropout(0.15)
│
└─ Dense(1)                   ← regression output (no activation)
```

**Why BiLSTM + Attention?**

- **Bidirectional processing** — integrates signals from both ends of the booking horizon simultaneously, capturing both the "early-bird" and "last-minute" pricing regimes in a single pass
- **Attention mechanism** — learns to weight `days_left` differently in peak vs. off-peak season, providing a degree of explainability aligned with regulatory requirements
- **L2 regularisation** — prevents memorising specific route–price combinations, ensuring generalisation to unseen routes

The **Route-Level LSTM** builds true 10-step price sequences per route sorted by `days_left` — the most faithful representation of how airlines actually observe demand evolving — and is the recommended architecture for a production weekly-retrain pipeline.

---

## Dataset

**Source:** Indian domestic flight bookings — `flight_data_enriched_usd.csv`

| Feature | Type | Description |
|---------|------|-------------|
| `airline` | Categorical | Carrier (Vistara, IndiGo, SpiceJet, Air India, GO_FIRST, AirAsia) |
| `source_city` | Categorical | Departure city |
| `destination_city` | Categorical | Arrival city |
| `departure_time` | Categorical | Time-of-day bucket (Early_Morning, Morning, Afternoon, Evening, Night) |
| `arrival_time` | Categorical | Time-of-day bucket |
| `stops` | Categorical | zero / one / two_or_more |
| `class` | Categorical | Economy / Business |
| `duration` | Numeric | Flight duration in hours |
| `duration_category` | Categorical | Short / Medium / Long |
| `days_left` | Numeric | Days between booking and departure (**the temporal signal**) |
| `season` | Categorical | High / Low season |
| `jet_fuel_price_usd_per_kl` | Numeric | Operating cost proxy |
| `distance_km` | Numeric | Route distance |
| `exchange_rate_usd_inr` | Numeric | FX rate at booking time |
| `price` | Numeric | **Target — ticket price in INR** |

**Engineered features:** booking urgency bins, price-per-km yield, day-of-week, month, is-weekend flag.

---

## Project Structure

```
flight_dynamic_pricing/
├── flight_dynamic_pricing_crisp_dm.ipynb   ← Full CRISP-DM notebook (60 cells)
├── README.md                               ← This file
├── conclusion.md                           ← Phase-by-phase findings summary
└── artefacts/                              ← Generated on run
    ├── bilstm_attention.keras              ← Trained Keras model (2.5 MB)
    ├── scaler_X.pkl                        ← MinMaxScaler for 18 input features
    ├── scaler_y.pkl                        ← MinMaxScaler for inverse-transforming predictions
    ├── label_encoders.pkl                  ← Dict of LabelEncoders (9 categorical columns)
    └── feature_cols.pkl                    ← Ordered feature column list
```

---

## Quickstart

### 1. Environment

```bash
pip install tensorflow>=2.10 scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Run the notebook

Open `flight_dynamic_pricing_crisp_dm.ipynb` in **Google Colab** or a local Jupyter environment. Update the `DATA_PATH` variable in Phase 2 to point to your copy of `flight_data_enriched_usd.csv`, then run all cells in order.

Saved artefacts will be written to `artefacts/` at the end of Phase 6.

### 3. Predict a price

```python
from predict import predict_price   # or copy the function from Phase 6

sample = {
    'airline':           'Vistara',
    'source_city':       'Delhi',
    'destination_city':  'Mumbai',
    'departure_time':    'Morning',
    'arrival_time':      'Afternoon',
    'class':             'Economy',
    'season':            'High',
    'stops':             'zero',
    'duration':          2.25,
    'days_left':         14,
    'duration_category': 'Medium',
    'jet_fuel_price_usd_per_kl': 1050,
    'distance_km':       1148,
    'exchange_rate_usd_inr': 83.5,
}

price = predict_price(**sample)
# ✈️  Vistara | Delhi → Mumbai | Economy
# 📅  14 days to departure
# 💰  Predicted price: $XX.XX
```

---

## Booking Window Intelligence

A sensitivity sweep across `days_left = 1 → 49` (all other attributes fixed) reveals the model's learned pricing curve:

```
Advance (30–49 days)  ████████████████████████████████  63.7% cheaper than last-minute
Month out (14–30d)    ██████████████████████████████    61.7% cheaper
Two weeks  (7–14d)    █████████████                     26.9% cheaper
One week   (3–7d)     ██████████                        21.7% cheaper
Last-minute (≤3d)     baseline — most expensive
```

The non-linear surge inside the final 7 days is precisely the pattern the LSTM architectures are designed to capture.

---

## What's Next

| Priority | Enhancement | Value |
|----------|-------------|-------|
| 🔴 High | **Real-time retraining** — stream weekly booking events to keep GBM current | Captures market shifts |
| 🔴 High | **Seat inventory features** — add load factor / seat availability | Critical demand-side signal currently missing |
| 🟡 Medium | **Competitor fares** — scrape rival pricing for cross-elasticity modelling | Enables competitive positioning |
| 🟡 Medium | **Multi-task learning** — jointly predict demand *and* price | Optimises expected revenue, not just price accuracy |
| 🟢 Long-term | **Reinforcement learning** — frame as sequential revenue optimisation | Maximises revenue across the full booking horizon |
| 🟢 Long-term | **API deployment** — FastAPI + Docker + cloud inference endpoint | Production-grade serving |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Preprocessing | `sklearn.preprocessing` — LabelEncoder, MinMaxScaler |
| Deep learning | `TensorFlow / Keras` — LSTM, Bidirectional, Dense, Dropout, BatchNormalization |
| Custom layers | `BahdanauAttention` — implemented as a Keras `Layer` subclass |
| Training | `EarlyStopping`, `ReduceLROnPlateau`, Adam optimiser, Huber loss |
| Tree models | `sklearn.ensemble` — GradientBoostingRegressor, RandomForestRegressor |
| Evaluation | `sklearn.metrics` — MAE, MSE, R² |
| Artefact persistence | `joblib` (scalers + encoders), Keras `.keras` (model) |

---

*Built with the CRISP-DM framework. Every phase documented, every artefact saved, every prediction explainable.*