# 🚀 CRM Analytics Lab

## 🏢 Business Problem

Customer churn is a major challenge for businesses—losing customers means losing revenue and growth opportunities. Traditional analytics often fail to provide actionable, real-time insights for retention and segmentation.

**This project solves:**
- 🔍 **Who is likely to churn?**
- 🧑‍🤝‍🧑 **How are customers segmented?**
- 📊 **What drives customer value and engagement?**
- 🤖 **How can business users interact with analytics easily?**

---

## 🛠️ Solution Approach

This solution is a **full-stack CRM analytics platform** built with Python, Flask, and modern data science libraries. It provides:

- **Automated Data Loading & Preprocessing**  
  Cleans, engineers, and scales features for robust modeling.

- **Exploratory Data Analysis & Visualization**  
  Generates insightful charts (e.g., churn by segment, spend distribution, clustering, sentiment, survival analysis) using `matplotlib` and `seaborn`.

- **Customer Segmentation**  
  - RFM (Recency, Frequency, Monetary) scoring
  - Clustering with `KMeans` and `DBSCAN`
  - Dimensionality reduction with `PCA`

- **Churn Prediction**  
  - Models: `LightGBM`, `RandomForest`, `XGBoost`
  - Handles class imbalance with `SMOTE`
  - Evaluates with cross-validation and confusion matrices

- **Explainable AI**  
  - `SHAP` for feature importance and model transparency

- **Time Series Forecasting**  
  - `ARIMA` for spend prediction

- **Sentiment Analysis**  
  - `TextBlob` for customer feedback

- **Interactive Dashboard**  
  - Built with Flask, HTML, CSS, and JavaScript
  - 🎛️ **KPI cards** for instant business metrics
  - 📈 **Dynamic chart grid** with slider to control chart visibility
  - 💬 **Chatbot** for business Q&A

- **REST API**  
  - Endpoints for health, summary, customer data, churn risk, prediction, and visualizations

## 📊 Visualizations Included

The dashboard and report include a rich set of visualizations to help you understand your customers and business trends:

- **Churn Rate by Customer Segment**  
  *Bar chart showing the proportion of churned customers in each segment.*

![churn_rate_by_segment](https://github.com/user-attachments/assets/80becf68-4d81-4b3f-9615-f3ea331a11ed)


- **Distribution of Total Spend**  
  *Histogram with KDE curve to visualize how much customers are spending.*

![total_spend_distribution](https://github.com/user-attachments/assets/12704579-f711-426a-82ae-ec110812ecb2)

- **Average Purchase Value by Reward Tier**  
  *Bar chart comparing average purchase values across different reward tiers.*

![avg_purchases_by_segment](https://github.com/user-attachments/assets/f8a22414-0ca1-41ab-a94c-8b26dd906265)

- **Average Number of Purchases by Customer Segment**  
  *Bar chart showing how frequently customers in each segment make purchases.*

![customer_sentiment](https://github.com/user-attachments/assets/b9b9ae75-00db-4496-a45f-970e2b9baf97)

- **Support Tickets by Churn Status**  
  *Boxplot comparing the number of support tickets for churned vs. non-churned customers.*

![support_tickets_by_churn](https://github.com/user-attachments/assets/58f95d94-54b9-4b0e-9708-22a808e9ef13)
 
- **Engagement Score Distribution by Segment**  
  *Violin plot showing the spread of engagement scores for each customer segment.*

![engagement_by_segment](https://github.com/user-attachments/assets/31ed7ed6-f90a-44bb-a2e1-6d21875c0d98)

- **Correlation Heatmap**  
  *Heatmap visualizing correlations between all numerical features.*

![correlation_heatmap](https://github.com/user-attachments/assets/bd24bdec-4ef5-4aa9-97aa-239822584cf0)

- **Customer Sentiment Distribution**  
  *Histogram of sentiment polarity scores from customer feedback.*

![customer_sentiment](https://github.com/user-attachments/assets/e8d083c0-0ef7-42e8-9094-aba2b8f458e0)

- **PCA Visualization of Customer Clusters**  
  *Scatter plot of customers in 2D PCA space, colored by RFM cluster.*

![pca_customer_clusters](https://github.com/user-attachments/assets/a4302fc5-9354-4f29-80cb-f76b46daede0)

- **DBSCAN Clustering**  
  *Scatter plot showing clusters found by DBSCAN based on spend, purchases, and engagement.*

![dbscan_clusters](https://github.com/user-attachments/assets/ca10e7cd-6da5-4b22-bbcc-fe8530501285)

- **Customer Retention Curve**  
  *Kaplan-Meier survival curve showing customer retention over time.*

![customer_retention_curve](https://github.com/user-attachments/assets/237fc02e-930d-45e0-abab-bf9012cf99d1)

- **Monthly Spending Forecast**  
  *Line plot with ARIMA forecast for future customer spending.*

![spending_forecast](https://github.com/user-attachments/assets/f1a4f19b-334a-4e97-8884-d163657d08b8)

- **SHAP Summary Plot**  
  *Feature importance plot explaining the LightGBM churn model.*

![shap_force_plot_improved](https://github.com/user-attachments/assets/93d7a93b-6045-439d-9b2f-3ec8ee6c5e57)

- **SHAP Feature Impact**  
  *Bar chart showing how each feature impacts churn prediction for a sample customer.*

![shap_feature_impact](https://github.com/user-attachments/assets/e9a1931e-a7d1-4f3a-b9b4-4a67000f90b4)

- **SHAP Force Plot**  
  *Step plot visualizing cumulative feature contributions to a churn prediction.*

![shap_force_plot_improved](https://github.com/user-attachments/assets/a6e12f67-3e97-4ace-a8bf-752e53baf463)

- **Confusion Matrices**  
  *Visual confusion matrices for LightGBM and RandomForest churn models.*

![confusion_matrix_lightgbm](https://github.com/user-attachments/assets/04235afe-f9be-4153-b3ea-2aeb4a3b79c9)

All charts are automatically generated and saved as images in the `charts/` folder, and are displayed in the dashboard and HTML report for easy exploration and business insight.

---

## 🖥️ How It Works

1. **Data Ingestion**  
   Loads customer data (CSV or synthetic), performs feature engineering, and handles missing values.

2. **Analytics & Modeling**  
   - Segments customers, predicts churn, explains results, and forecasts trends.
   - Saves all visualizations as PNGs in the `charts/` folder.

3. **Dashboard**  
   - Access at [http://localhost:5000/dashboard](http://localhost:5000/dashboard)
   - View KPIs, charts, and interact with the chatbot.

4. **API**  
   - `/summary` – KPIs  
   - `/customers` – Customer data  
   - `/churn-risk` – Churn by segment  
   - `/predict-churn` – Predict churn for a customer  
   - `/visualizations` – Chart URLs

5. **Chatbot**  
   - Click "💬 Open CRM Chatbot" in the dashboard for instant Q&A.

---

## 🧩 Key Technologies

- **Python**: pandas, numpy, scikit-learn, seaborn, matplotlib, xgboost, lightgbm, shap, lifelines, textblob, imblearn
- **Flask**: API & dashboard
- **HTML/CSS/JS**: Responsive, modern UI
- **Logging**: For traceability and debugging

---

## 🌟 Value to Business

- 📉 **Reduce churn** with predictive analytics
- 🎯 **Target marketing** via segmentation
- 😊 **Boost satisfaction** with sentiment & engagement insights
- 🧑‍💼 **Empower users** with dashboard & chatbot
- 🧠 **Explainable AI** for trust and transparency

---

## 🚦 Quick Start

```bash
# 1. Install dependencies
conda env create -f environment.yml
# or
pip install -r requirements.txt

# 2. Run the app
python Crm_ml.py

# 3. Open your browser
http://localhost:5000/dashboard
```

---

## 📝 Customization

- Add your own customer CSV data
- Extend dashboard with new KPIs or charts
- Integrate with other systems via the API

---

## 📄 License

MIT License

---

> Made with ❤️ for data-driven CRM excellence!
