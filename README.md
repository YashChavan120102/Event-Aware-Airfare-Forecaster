# ✈️ Event-Aware Airfare Forecaster (EAAF)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-LightGBM-green)
![App](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

------------------------------------------------------------------------

## 🚀 Overview

**What if flight booking felt less like gambling---and more like
investing?**

The **Event-Aware Airfare Forecaster (EAAF)** is an end-to-end machine
learning system that predicts airfare trends and provides **Buy vs Wait
recommendations** using real-world contextual signals like:

-   📅 Holidays
-   🎉 Local Events
-   🌦️ Weather
-   📊 Booking behavior

------------------------------------------------------------------------

## 🎯 Business Problem

Flight booking platforms today: - Show only current prices\
- Ignore demand signals
- Lack historical intelligence

👉 Result: - Users overpay
- Poor booking timing
- High uncertainty

------------------------------------------------------------------------

## 💡 Solution

EAAF answers one key question:

### 👉 **Should I book now, or wait?**

### ✅ Features

-   📈 Price trend forecasting
-   🧠 Buy vs Wait recommendation
-   📅 Optimal booking date suggestion
-   🔍 Explainable predictions

------------------------------------------------------------------------

## 🏗️ System Architecture

    Data Sources → Feature Engineering → ML Model → Prediction → Recommendation Engine → Streamlit Dashboard

------------------------------------------------------------------------

## 🔄 End-to-End Pipeline

### 1️⃣ Data Collection

-   Flight fare dataset (synthetic + historical)
-   Holiday data (python-holidays)
-   Event data (Ticketmaster API)
-   Weather data (Meteostat API)

------------------------------------------------------------------------

### 2️⃣ Feature Engineering

-   Holiday Proximity Score
-   Event Intensity Index
-   Weather Impact Features
-   Seasonality & Weekends
-   Days-to-Departure

------------------------------------------------------------------------

### 3️⃣ Model

-   LightGBM (Gradient Boosting)
-   Regression-based forecasting

------------------------------------------------------------------------

### 4️⃣ Recommendation Engine

-   BUY → Prices expected to rise
-   WAIT → Price drop predicted
-   Suggests optimal booking date

------------------------------------------------------------------------

### 5️⃣ Dashboard (Streamlit)

Interactive app to: - Input travel details
- View predictions
- Get recommendations
- Understand reasoning

------------------------------------------------------------------------

## 📸 Demo

> 📌<img width="1897" height="906" alt="EAAF" src="https://github.com/user-attachments/assets/c6aa2863-dc5d-44aa-907b-cb0d4c3f961f" />


------------------------------------------------------------------------

## 📊 Example Output

  Route            Insight
  ---------------- --------------
  Berlin → Paris   WAIT
  Event detected   Demand surge
  Recommendation   Book later

------------------------------------------------------------------------

## 🧠 Key Highlights

✔️ Combines ML + real-world signals
✔️ Moves from prediction → decision intelligence
✔️ Fully end-to-end pipeline
✔️ Business-focused ML application

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   Pandas, NumPy
-   LightGBM
-   Requests, Dotenv
-   Streamlit

------------------------------------------------------------------------

## 📈 Impact

### 👤 Users

-   Better booking decisions
-   Reduced costs

### 🏢 Businesses

-   Pricing insights
-   Travel optimization

------------------------------------------------------------------------

## 🔐 Data Note

Synthetic dataset used due to lack of historical airfare snapshots in
public APIs.
Real-world signals integrated for realism.

------------------------------------------------------------------------

## 🔮 Future Work

-   SHAP explainability
-   LSTM / Prophet models
-   Cloud deployment
-   Real-time pipelines

------------------------------------------------------------------------

## 🤝 Let's Connect

If you're interested in: - Travel Tech
- Data Products
- Machine Learning

Feel free to connect!

------------------------------------------------------------------------

## 👨‍💻 Author

**Yash Chavan**
MSc Data Science | Berlin
