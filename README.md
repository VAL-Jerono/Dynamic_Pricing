# ✈️ Flight Ticket Dynamic Pricing with LSTM

> *"Why does the same seat cost ₹3,500 today but ₹9,200 tomorrow?"*  
> This project builds a machine learning engine to answer — and predict — exactly that.

---

## The Story

Every time you search for a flight, an invisible pricing algorithm is watching — tracking demand surges, days to departure, seat availability, and route competition. Prices don't just change daily; they change by the hour. Airlines call it **dynamic pricing**, and it's one of the most profitable levers in the industry.

This project reverse-engineers that logic. Given a flight's attributes — airline, route, cabin class, number of stops, and critically, **how many days are left before departure** — we train a deep learning model to predict the ticket price with high accuracy. The temporal nature of the problem makes it a natural fit for an **LSTM (Long Short-Term Memory)** network, which excels at learning patterns that evolve over time.

The entire workflow follows the **CRISP-DM framework**, ensuring the work is structured, reproducible, and deployment-ready.

---

## CRISP-DM at a Glance

| Phase | What We Did |
|---|---|
| 1. Business Understanding | Framed ticket pricing as a regression problem anchored on the `days_left` signal |
| 2. Data Understanding | Explored price distributions, airline premiums, stop penalties, and temporal trends |
| 3. Data Preparation | Label-encoded categoricals, MinMax-scaled features and target, reshaped for LSTM |
| 4. Modeling | Built a 2-layer stacked LSTM with BatchNorm + Dropout, trained with adaptive callbacks |
| 5. Evaluation | Assessed MAE, RMSE, MAPE, and R² on a held-out test set; visualized residuals |
| 6. Deployment | Saved model + all artifacts; built a `predict_price()` function + sensitivity analysis |

---

## Phase 1 — Business Understanding

**The core question:** Can a model learn the non-linear relationship between flight attributes and price — including the urgency premium airlines charge as departure approaches?

**Business questions that drove the analysis:**

- How do ticket prices evolve as departure day approaches (`days_left`)?
- Which airlines and routes command the highest premiums?
- Does cabin class and number of stops meaningfully shift price?
- Can an LSTM learn these temporal pricing patterns well enough to power real predictions?

**Target variable:** `price` (continuous — regression task, prices in INR)  
**Key temporal driver:** `days_left` — the single most important signal, and the reason LSTM is the right architecture.

---

## Phase 2 — Data Understanding

The dataset covers Indian domestic flight bookings with the following features:

| Feature | Type | Description |
|---|---|---|
| `airline` | Categorical | Carrier (e.g., Vistara, IndiGo, Air India) |
| `source_city` | Categorical | Departure city |
| `destination_city` | Categorical | Arrival city |
| `departure_time` | Categorical | Time-of-day bucket (Morning, Evening, etc.) |
| `arrival_time` | Categorical | Time-of-day bucket |
| `stops` | Categorical | zero / one / two\_or\_more |
| `class` | Categorical | Economy / Business |
| `duration` | Numeric | Flight duration in hours |
| `duration_category` | Categorical | Short / Medium / Long bucketing of duration |
| `days_left` | Numeric | Days between booking and departure (**the temporal signal**) |
| `price` | Numeric | **Target — ticket price in INR** |

**Key findings from exploratory analysis:**

- **Price distribution** is right-skewed — a long tail of premium-priced Business class tickets pulls the mean well above the median. Most Economy tickets cluster in a narrower band.
- **Airline premium:** Vistara and Air India consistently command higher fares. Budget carriers like IndiGo and SpiceJet anchor the lower end.
- **The `days_left` curve:** Average prices are highest far from departure (early bookers pay a premium on some routes), dip in the middle, then **surge sharply in the final 7–14 days** as last-minute travelers compete for remaining seats. This non-linear pattern is precisely what the LSTM is designed to capture.
- **Stops penalty:** Direct flights (`zero`) actually price higher on premium routes — convenience commands a premium. Multi-stop routes are cheaper but not uniformly so.
- **Class gap:** Business class tickets are significantly more expensive, making `class` one of the strongest predictors in the feature set.
- **Duration vs Price:** A loose positive correlation — longer flights tend to cost more, but route and airline matter more than raw duration.

---

## Phase 3 — Data Preparation

Raw flight records don't speak the language of neural networks. This phase transforms them.

**Label Encoding**  
All categorical columns (`airline`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class`, `duration_category`) were label-encoded using `sklearn.preprocessing.LabelEncoder`. Crucially, each encoder was saved separately — they're required artifacts at prediction time to transform new inputs consistently.

The `flight` column (a route code, not a predictive feature) was dropped.

**Feature Scaling**  
Two `MinMaxScaler` instances were fitted independently:

- `feature_scaler` — scales all 10 input features to [0, 1]
- `price_scaler` — scales the target (`price`) to [0, 1]

Scaling the target is non-negotiable for LSTM stability. Without it, the model's output layer must learn to produce values in the tens of thousands (INR), which creates gradient issues. Scaling brings the problem into a numerically well-behaved range and both scalers are saved for inverse-transform at inference time.

**Reshaping for LSTM**  
LSTMs expect input of shape `(samples, timesteps, features)`. Since each flight record is an independent booking (not part of a named time-series), `timesteps = 1` was used. The LSTM still extracts non-linear feature interactions — it just doesn't sequence across multiple past bookings per route. For a true time-series formulation, one would group by route and sort by `days_left` descending.

**Train / Test Split:** 80/20 stratified random split (`random_state=42`).

---

## Phase 4 — Modeling

### Architecture

A 2-layer stacked LSTM was chosen for its ability to capture both low-level feature interactions (layer 1) and higher-order temporal abstractions (layer 2).

```
Input: (batch, 1, 10 features)
│
├─ LSTM(128, return_sequences=True)
├─ Dropout(0.3)
├─ BatchNormalization()
│
├─ LSTM(64, return_sequences=False)
├─ Dropout(0.2)
├─ BatchNormalization()
│
├─ Dense(32, activation='relu')
├─ Dropout(0.1)
│
└─ Dense(1)           ← Regression output (no activation)
```

**Design choices:**

- `return_sequences=True` on the first LSTM passes the full hidden state sequence to the second layer — enabling the stacking.
- `BatchNormalization` after each LSTM layer stabilizes training and reduces sensitivity to learning rate choice.
- `Dropout` at 0.3 → 0.2 → 0.1 (decreasing towards the output) guards against overfitting without over-regularizing the final representation.
- Final `Dense(1)` has no activation — this is a regression task and we want unbounded real-valued output.

**Loss function:** Mean Squared Error (MSE) — penalises large errors more heavily, appropriate for a pricing problem where a ₹5,000 miss is much worse than a ₹500 miss.

**Optimiser:** Adam with `lr=0.001`.

### Training

```python
history = model.fit(
    X_train_lstm, y_train,
    epochs=100,
    batch_size=256,
    validation_split=0.15,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
               ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)]
)
```

Two callbacks governed training:

- **EarlyStopping** (patience=10): halts training if validation loss stops improving for 10 consecutive epochs and restores the best weights seen during the run — not the final epoch's weights.
- **ReduceLROnPlateau** (factor=0.5, patience=5): halves the learning rate when the model plateaus, allowing finer convergence without a hand-tuned schedule.

Training curves (Loss and MAE plotted per epoch) confirmed healthy convergence — train and validation curves tracking closely without divergence, indicating the regularisation was appropriate.

---

## Phase 5 — Evaluation

Predictions were inverse-transformed back to INR before computing metrics, so all reported numbers are in the original price scale — interpretable as actual rupee errors.

| Metric | Meaning |
|---|---|
| **MAE** | Average absolute error in INR — the everyday "how far off" measure |
| **RMSE** | Root Mean Squared Error — penalises large misses more than MAE |
| **MAPE** | Mean Absolute Percentage Error — relative error, scale-independent |
| **R²** | Proportion of price variance explained by the model |

**Visualisations produced:**

1. **Actual vs Predicted scatter plot** — points cluster along the diagonal, confirming the model tracks true prices across the full range. Business class outliers (high actual prices) show slightly more spread, as expected.
2. **Residuals histogram** — approximately centred on zero with a roughly normal shape, indicating no systematic bias. The model is not consistently over- or under-pricing.
3. **Average Actual vs Predicted Price by Days Left** — the model's temporal intelligence on display. Predicted average prices mirror the actual `days_left` pricing curve, including the late-booking surge. This is the proof that the LSTM captured the dynamic pricing signal.

---

## Phase 6 — Deployment

### Saved Artifacts

All artifacts needed for inference are saved as a portable bundle:

```
dynamic_pricing_model/
├── lstm_pricing_model.h5     ← Trained Keras model
├── feature_scaler.pkl        ← MinMaxScaler for input features
├── price_scaler.pkl          ← MinMaxScaler for inverse-transforming predictions
└── label_encoders.pkl        ← Dict of LabelEncoders, one per categorical column
```

### Prediction Function

```python
def predict_price(flight_details: dict) -> float:
    """
    Predict ticket price for a new flight.

    Parameters
    ----------
    flight_details : dict
        Keys: airline, source_city, departure_time, stops, arrival_time,
              destination_city, class, duration, days_left, duration_category

    Returns
    -------
    float : Predicted price in INR
    """
```

The function handles the full inference pipeline: encoding categoricals with saved encoders → scaling with `feature_scaler` → reshaping to `(1, 1, 10)` → model forward pass → inverse-transform via `price_scaler` → return INR price.

**Example:**

```python
sample_flight = {
    'airline':           'Vistara',
    'source_city':       'Delhi',
    'departure_time':    'Morning',
    'stops':             'zero',
    'arrival_time':      'Afternoon',
    'destination_city':  'Mumbai',
    'class':             'Economy',
    'duration':          2.25,
    'days_left':         15,
    'duration_category': 'Medium'
}

predicted = predict_price(sample_flight)
# ✈️  Vistara | Delhi → Mumbai
# 📅  Days to departure: 15
# 💰  Predicted Price: ₹X,XXX.XX
```

### Price Sensitivity Analysis

A sensitivity sweep across `days_left = 1 to 49` (holding all other attributes fixed) produces the model's learned pricing curve for any given route. The chart — plotted with x-axis inverted so left = far from departure, right = day of travel — reveals the model's internalized version of airline dynamic pricing: prices are moderate far out, compress slightly in the mid-range, then spike sharply in the final week. The vertical markers at 7 and 14 days highlight the critical pricing windows.

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Preprocessing | `sklearn.preprocessing` (LabelEncoder, MinMaxScaler) |
| Deep learning | `TensorFlow / Keras` (Sequential, LSTM, Dense, Dropout, BatchNormalization) |
| Training callbacks | `EarlyStopping`, `ReduceLROnPlateau` |
| Evaluation | `sklearn.metrics` (MAE, MSE, R²) |
| Artifact persistence | `joblib` (scalers + encoders), Keras `.h5` (model) |

---

## Project Structure

```
flight_dynamic_pricing_lstm/
├── flight_dynamic_pricing_lstm.ipynb   ← Full CRISP-DM notebook
├── README.md                           ← This file
└── dynamic_pricing_model/              ← Saved artifacts (generated on run)
    ├── lstm_pricing_model.h5
    ├── feature_scaler.pkl
    ├── price_scaler.pkl
    └── label_encoders.pkl
```

---

## Reproducing the Results

1. Clone the repo and open the notebook in Google Colab or a local Jupyter environment.
2. Place `flight_data_cleaned.csv` at the path referenced in Phase 2 (or update the `pd.read_csv` path to match your setup).
3. Run all cells in order — the notebook is self-contained and follows the CRISP-DM phases sequentially.
4. Saved artifacts will be written to `dynamic_pricing_model/` at the end of Phase 6.

**Requirements:**

```
tensorflow>=2.10
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

---

## What's Next

The current model treats each flight record as independent. The natural evolution is to make the time-series structure explicit:

- **True sequential modeling** — group records by route, sort by `days_left` descending, and feed actual price sequences into the LSTM rather than individual snapshots.
- **Feature engineering** — add day-of-week, public holidays, school vacation flags, and route-level demand proxies (historical booking volume).
- **Architecture experiments** — Transformer-based models (e.g., Temporal Fusion Transformer) have shown strong performance on tabular time-series and are worth benchmarking against the LSTM baseline.
- **API deployment** — wrap `predict_price()` in a FastAPI endpoint, containerise with Docker, and deploy for real-time pricing inference.
- **Confidence intervals** — implement Monte Carlo Dropout to return price ranges rather than point estimates, giving downstream consumers a sense of prediction uncertainty.

---

*Built with the CRISP-DM framework. Every phase documented, every artifact saved, every prediction explainable.*