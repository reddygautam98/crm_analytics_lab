# 🧠 CRM Analytics Engine with ML & API Interface

Welcome to the **CRM Analytics Engine**, a full-stack intelligent analytics and automation framework for Customer Relationship Management (CRM). This end-to-end solution integrates **machine learning**, **data visualization**, and **RESTful API services** to deliver actionable insights into customer behavior, churn prediction, segmentation, and engagement scoring.

---

## 🔍 Features

### 📊 Data Ingestion & Preprocessing
- Synthetic and real CSV ingestion with schema validation
- Automated data cleaning, outlier detection (IsolationForest), and feature engineering
- Intelligent imputation and encoding using `pandas`, `scikit-learn`, and `imblearn`

### 🧠 Machine Learning Pipeline
- Churn prediction with **LightGBM**, **Random Forest**, and **XGBoost**
- Model evaluation: `confusion_matrix`, `classification_report`, `accuracy_score`
- Explainability powered by **SHAP**

### 🎯 Customer Intelligence
- **RFM Segmentation** using quantile-based binning and robust fallback methods
- **KMeans** and **DBSCAN** clustering for customer stratification
- **Sentiment analysis** of feedback using `TextBlob`

### 📈 Forecasting & Survival Analysis
- Time-series forecasting using **ARIMA** for spend trends
- **Kaplan-Meier Survival Curves** for customer retention analytics

### 🌍 Geo-Visualization
- Interactive customer location map using **Folium**

### 🖼️ Rich Visual Reporting
- Comprehensive visual suite powered by `matplotlib`, `seaborn`, and `folium`
- 📌 **PCA Clustering Plot**: Projects customer data into 2D space using Principal Component Analysis for intuitive cluster visualization
- 📌 **DBSCAN Clustering**: Reveals natural groupings based on density and spending behavior
- 📌 **Sentiment Distribution Histogram**: Highlights the polarity of customer feedback
- 📌 **Customer Retention Curve**: Survival plot estimating the probability of customer retention over time
- 📌 **Spending Forecast**: Time-series projection of monthly revenue using ARIMA model
- 📌 **SHAP Summary Plot**: Interprets model output by ranking feature importance
- 📌 **Customer Geo Map**: Interactive marker-based map showing customer locations with segment and churn info
- All visuals are compiled into an auto-generated HTML report for convenient sharing
- Auto-generated charts: PCA clusters, churn heatmaps, sentiment histograms
- Generates a responsive **HTML report** with all metrics and visuals

### 🛠️ RESTful API (Flask)
Expose your ML capabilities with endpoints:
- `/health` - Health check
- `/summary` - CRM KPIs
- `/customers` - Customer dataset
- `/churn-risk` - Segment-wise churn probability
- `/predict-churn` - Predict churn for new input
- `/chatbot` - Basic NLP chatbot for FAQs

---

## 🚀 Quick Start

### 🔧 Prerequisites
Ensure the following are installed:
```bash
pip install -r requirements.txt
```

### 🏃 Run the Application
```bash
python Crm_ml.py
```

> Outputs will be logged to versioned files in `/logs` and charts saved in `/charts`.

---

## 📂 Project Structure
```
.
├── Crm_ml.py               # Main script
├── charts/                 # Generated visualizations (PNG, HTML)
├── logs/                   # Timestamped execution logs
├── synthetic_customer_data_500.csv  # Input dataset
└── report-*.html           # Auto-generated dashboard reports
```

---

## 💡 Highlights

| Module        | Tools & Libraries                            |
|---------------|-----------------------------------------------|
| ML Models     | LightGBM, XGBoost, RandomForest               |
| Preprocessing | pandas, scikit-learn, imblearn (SMOTE)        |
| NLP           | TextBlob                                      |
| Forecasting   | ARIMA (statsmodels)                           |
| Clustering    | KMeans, DBSCAN                                |
| Explainability| SHAP                                          |
| Visualization | matplotlib, seaborn, folium                   |
| API           | Flask                                         |

---

## 📬 Example Churn Prediction API Call
```bash
curl -X POST http://localhost:5000/predict-churn \
-H "Content-Type: application/json" \
-d '{
  "Age": 30,
  "TotalSpend": 1200.0,
  "NumberOfPurchases": 10,
  "SupportTickets": 1,
  "DaysSinceLastLogin": 15,
  "AvgPurchaseValue": 120.0,
  "PurchaseFrequency": 0.67,
  "EngagementScore": 0.85,
  "SupportTicketRatio": 0.1
}'
```

---

## 🧪 Model Performance (Cross-Validated)

| Model         | Accuracy (± StdDev) |
|---------------|----------------------|
| LightGBM      | ~**87% ± 2%**        |
| Random Forest | ~**85% ± 3%**        |
| XGBoost       | ~**83%**             |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🤝 Contributions

Pull requests, feedback, and feature suggestions are welcome!

---

## 📨 Contact

Developed by [Gautam]. For questions, reach out via [LinkedIn](https://www.linkedin.com/in/gautam-reddy-359594261/) or open an issue.
