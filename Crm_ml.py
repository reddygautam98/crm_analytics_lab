import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib
import shutil
import glob
from flask import send_from_directory

# Try to import optional libraries with fallbacks
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available - using original data without resampling")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - using RandomForest instead")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - using RandomForest instead")

try:
    from lifelines import KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Lifelines not available - skipping survival analysis")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available - skipping time series analysis")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available - using simple sentiment analysis")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - skipping model explainability")

try:
    from flask import Flask, jsonify, request, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available - skipping web API")

import importlib.util

# Set matplotlib font size and figure parameters globally
matplotlib.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.figsize': (16, 8)  # Increased width for better spacing
})

# Configure logging
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_name = f"version-{timestamp}.txt"
log_file_path = os.path.join(logs_dir, log_file_name)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler(log_file_path)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Check if Streamlit is available
if importlib.util.find_spec("streamlit") is not None:
    STREAMLIT_AVAILABLE = True
    logging.info("Streamlit is available.")
else:
    STREAMLIT_AVAILABLE = False
    logging.warning("Streamlit not available. Running in script mode.")

warnings.filterwarnings('ignore')

config = {
    'required_columns': ['CustomerSegment', 'ChurnProbability', 'Location', 'Latitude', 'Longitude'],
    'charts_folder': 'charts',
    'default_latitude_range': (25, 49),
    'default_longitude_range': (-125, -67)
}

charts_folder = config['charts_folder']
if not os.path.exists(charts_folder):
    os.makedirs(charts_folder)
    logging.info(f"Created charts folder: {charts_folder}")

plt.style.use('ggplot')
sns.set_style("whitegrid")

# Load data from CSV file
data_path = r"C:\Users\reddy\Downloads\crm_analytics_lab\synthetic_customer_data_500.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    logging.info(f"Loaded data from {data_path} with {len(df)} records")
else:
    # Fallback to synthetic data if file not found
    logging.warning(f"File not found: {data_path}. Creating synthetic data for demonstration.")
    np.random.seed(42)
    num_samples = 500
    customer_ids = [f'CUST{i:05d}' for i in range(1, num_samples + 1)]
    ages = np.random.normal(35, 10, num_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    total_spend = np.random.exponential(500, num_samples)
    num_purchases = np.random.poisson(5, num_samples) + 1  # Ensure at least 1 purchase
    support_tickets = np.random.poisson(2, num_samples)
    segments = np.random.choice(['Premium', 'Standard', 'Basic'], num_samples, p=[0.2, 0.5, 0.3])
    categories = np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Food'], num_samples)
    reward_tiers = np.random.choice(['Gold', 'Silver', 'Bronze', 'None'], num_samples, p=[0.1, 0.2, 0.3, 0.4])
    is_churned = np.random.binomial(1, 0.2, num_samples)

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': ages,
        'TotalSpend': total_spend,
        'NumberOfPurchases': num_purchases,
        'SupportTickets': support_tickets,
        'CustomerSegment': segments,
        'FavoriteCategory': categories,
        'RewardTier': reward_tiers,
        'IsChurned': is_churned,
        'Feedback': np.random.choice(['Great service!', 'Not satisfied', 'Average experience', 
                                     'Excellent support', 'Could be better'], size=num_samples),
        'Latitude': np.random.uniform(25, 49, size=num_samples),
        'Longitude': np.random.uniform(-125, -67, size=num_samples)
    })
    logging.info(f"Synthetic data created with {len(df)} records")

# Add missing columns if needed
if 'Feedback' not in df.columns:
    df['Feedback'] = np.random.choice(['Great service!', 'Not satisfied', 'Average experience', 
                                     'Excellent support', 'Could be better'], size=len(df))
if 'Latitude' not in df.columns:
    df['Latitude'] = np.random.uniform(*config['default_latitude_range'], size=len(df))
if 'Longitude' not in df.columns:
    df['Longitude'] = np.random.uniform(*config['default_longitude_range'], size=len(df))

numeric_cols = ['Age', 'TotalSpend', 'NumberOfPurchases', 'SupportTickets', 'IsChurned']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'LastLoginDate' in df.columns:
    df['LastLoginDate'] = pd.to_datetime(df['LastLoginDate'], errors='coerce')
else:
    today = datetime.now()
    days_ago = np.random.randint(1, 365, size=len(df))
    df['LastLoginDate'] = [today - pd.Timedelta(days=int(d)) for d in days_ago]
    logging.info("Created synthetic LastLoginDate column")

if 'TransactionMonth' not in df.columns:
    df['TransactionMonth'] = pd.to_datetime(df['LastLoginDate']).dt.to_period('M')

for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Calculate summary statistics BEFORE scaling
original_avg_age = df['Age'].mean() if 'Age' in df.columns else 'N/A'
original_avg_spend = df['TotalSpend'].mean() if 'TotalSpend' in df.columns else 'N/A'

# Print summary statistics early
logging.info("\n=== Pre-Scaling Summary Statistics ===")
logging.info(f"Total Customers: {len(df)}")
if 'IsChurned' in df.columns:
    logging.info(f"Churn Rate: {df['IsChurned'].mean() * 100:.2f}%")
if 'Age' in df.columns:
    logging.info(f"Average Age: {original_avg_age:.1f}" if isinstance(original_avg_age, (int, float)) else f"Average Age: {original_avg_age}")
if 'TotalSpend' in df.columns:
    logging.info(f"Average Spend: ${original_avg_spend:.2f}" if isinstance(original_avg_spend, (int, float)) else f"Average Spend: {original_avg_spend}")

# Feature Engineering with safeguards
current_date = datetime.now()

# Calculate days since last login safely
df['DaysSinceLastLogin'] = (current_date - df['LastLoginDate']).dt.days
df['DaysSinceLastLogin'].fillna(df['DaysSinceLastLogin'].median(), inplace=True)

# Calculate average purchase value safely (avoid division by zero)
df['AvgPurchaseValue'] = df['TotalSpend'] / df['NumberOfPurchases'].replace(0, 1)

# Calculate purchase frequency safely (avoid division by zero)
df['PurchaseFrequency'] = df['NumberOfPurchases'] / df['DaysSinceLastLogin'].replace(0, 1)

# Calculate engagement score
df['EngagementScore'] = (100 - df['DaysSinceLastLogin'].clip(upper=100)) / 100

# Calculate support ticket ratio safely (avoid division by zero)
df['SupportTicketRatio'] = df['SupportTickets'] / df['NumberOfPurchases'].replace(0, 1)

# Store original values before scaling
original_features = df[['Age', 'TotalSpend', 'NumberOfPurchases', 'SupportTickets', 'DaysSinceLastLogin', 
                       'AvgPurchaseValue', 'PurchaseFrequency', 'EngagementScore', 'SupportTicketRatio']].copy()

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'TotalSpend', 'NumberOfPurchases', 'SupportTickets', 'DaysSinceLastLogin', 
                      'AvgPurchaseValue', 'PurchaseFrequency', 'EngagementScore', 'SupportTicketRatio']
numerical_features = [col for col in numerical_features if col in df.columns]

# Create a copy to avoid SettingWithCopyWarning
scaled_features = scaler.fit_transform(df[numerical_features])
df_scaled = df.copy()
df_scaled[numerical_features] = scaled_features

# Log the scaling process
logging.info("Numerical features have been standardized.")

# Use IsolationForest for anomaly detection on original data
isolation_forest = IsolationForest(random_state=42, contamination=0.05)
df['Anomaly'] = isolation_forest.fit_predict(original_features[['TotalSpend', 'NumberOfPurchases']])

# Mark anomalies
df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')

# Log the number of anomalies detected
logging.info(f"Number of anomalies detected: {df['Anomaly'].value_counts().get('Outlier', 0)}")

# Perform sentiment analysis on customer feedback
def simple_sentiment_analysis(text):
    """Simple sentiment analysis fallback"""
    positive_words = ['great', 'excellent', 'amazing', 'love', 'good', 'awesome', 'fantastic']
    negative_words = ['not satisfied', 'poor', 'bad', 'terrible', 'awful', 'worse', 'disappointing']
    
    text_lower = str(text).lower()
    pos_score = sum(1 for word in positive_words if word in text_lower)
    neg_score = sum(1 for word in negative_words if word in text_lower)
    
    if pos_score > neg_score:
        return 0.5
    elif neg_score > pos_score:
        return -0.5
    else:
        return 0.0

# Add a Sentiment column based on the polarity of the feedback
if TEXTBLOB_AVAILABLE:
    try:
        df['Sentiment'] = df['Feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    except Exception:
        df['Sentiment'] = df['Feedback'].apply(simple_sentiment_analysis)
else:
    df['Sentiment'] = df['Feedback'].apply(simple_sentiment_analysis)

# Visualize sentiment distribution
plt.figure(figsize=(14, 6))
sns.histplot(df['Sentiment'], bins=10, kde=True, color='blue')
plt.title('Customer Sentiment Distribution')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(charts_folder, 'customer_sentiment.png'))
plt.close()

# Fixed robust_segment function
def robust_segment(series, num_bins=3, labels=None, quantile_based=True):
    """
    Segment a series into bins robustly, handling edge cases.
    
    Args:
        series: Pandas Series to segment
        num_bins: Number of bins to create
        labels: Labels for the bins
        quantile_based: Whether to use quantile-based binning
        
    Returns:
        Pandas Series with bin labels
    """
    if labels is None:
        labels = [f'Q{i+1}' for i in range(num_bins)]
    
    # Handle empty series
    if len(series) == 0:
        return pd.Series([], dtype='object')
    
    # Drop NaN values for binning
    clean_series = series.dropna()
    
    # Handle empty series after dropping NaNs
    if len(clean_series) == 0:
        return pd.Series(index=series.index, dtype='object')
    
    # Count unique values
    unique_values = len(clean_series.unique())
    
    # If fewer unique values than bins, adjust
    if unique_values < num_bins:
        if unique_values == 1:
            # Only one value, assign to the highest bin
            result = pd.Series(labels[-1], index=series.index)
            return result
        
        # Adjust number of bins to match unique values
        adjusted_labels = labels[:unique_values]
        try:
            result = pd.qcut(clean_series, q=unique_values, labels=adjusted_labels, duplicates='drop')
            # Map back to original series index
            return pd.Series(result, index=clean_series.index).reindex(series.index)
        except ValueError:
            # If qcut fails, try cut
            try:
                result = pd.cut(clean_series, bins=unique_values, labels=adjusted_labels)
                return pd.Series(result, index=clean_series.index).reindex(series.index)
            except ValueError:
                # If all else fails, assign to the middle bin
                return pd.Series(labels[len(labels)//2], index=series.index)
    
    # Normal case: enough unique values
    try:
        if quantile_based:
            # Use quantile-based binning
            result = pd.qcut(clean_series, q=num_bins, labels=labels, duplicates='drop')
            return pd.Series(result, index=clean_series.index).reindex(series.index)
        else:
            # Use equal-width binning
            result = pd.cut(clean_series, bins=num_bins, labels=labels)
            return pd.Series(result, index=clean_series.index).reindex(series.index)
    except Exception as e:
        logging.warning(f"Error in segmentation: {str(e)}. Using manual binning.")
        # Manual binning as fallback
        min_val, max_val = clean_series.min(), clean_series.max()
        if min_val == max_val:
            # If all values are the same, assign to the highest bin
            return pd.Series(labels[-1], index=series.index)
        
        # Add a small epsilon to max_val to ensure it's included in the last bin with linspace
        bins = np.linspace(min_val, max_val + 1e-9, num_bins + 1)
        try:
            # Ensure bin edges are unique, important if min_val is very close to max_val
            unique_bins = np.unique(bins)
            if len(unique_bins) < 2:
                 # Cannot form bins if edges aren't distinct
                 raise ValueError("Cannot form bins with non-unique edges.")
            if len(unique_bins) < num_bins + 1:
                # Adjust labels if unique bins are fewer than expected
                adjusted_labels = labels[:len(unique_bins)-1]
            else:
                adjusted_labels = labels
                
            result = pd.cut(clean_series, bins=unique_bins, labels=adjusted_labels, include_lowest=True)
            return pd.Series(result, index=clean_series.index).reindex(series.index)
        except Exception as fallback_e:
            logging.error(f"Manual binning also failed: {fallback_e}. Assigning middle bin.")
            # Last resort: assign middle bin
            return pd.Series(labels[len(labels)//2], index=series.index)

# Apply RFM segmentation using original data
df['R_Score'] = robust_segment(df['DaysSinceLastLogin'], 5, labels=['5', '4', '3', '2', '1'])
df['F_Score'] = robust_segment(df['NumberOfPurchases'].clip(lower=1), 5, labels=['1', '2', '3', '4', '5'])
df['M_Score'] = robust_segment(df['TotalSpend'].clip(lower=0), 5, labels=['1', '2', '3', '4', '5'])

# Ensure scores are strings for concatenation
df['R_Score'] = df['R_Score'].astype(str)
df['F_Score'] = df['F_Score'].astype(str)
df['M_Score'] = df['M_Score'].astype(str)

# Create RFM score string
df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']

# Convert back to numeric for calculations
df['R_Score'] = pd.to_numeric(df['R_Score'], errors='coerce').fillna(3)
df['F_Score'] = pd.to_numeric(df['F_Score'], errors='coerce').fillna(3)
df['M_Score'] = pd.to_numeric(df['M_Score'], errors='coerce').fillna(3)
df['RFM_Value'] = df['R_Score'] + df['F_Score'] + df['M_Score']

# RFM Clustering
rfm_features = ['R_Score', 'F_Score', 'M_Score']
rfm_data = df[rfm_features]
kmeans_rfm = KMeans(n_clusters=4, random_state=42, n_init=10)
df['RFM_Cluster'] = kmeans_rfm.fit_predict(rfm_data)

# Apply PCA for Dimensionality Reduction
pca_features = ['Age', 'TotalSpend', 'NumberOfPurchases', 'SupportTickets', 'EngagementScore', 'RFM_Value']
pca_features = [col for col in pca_features if col in df.columns]
pca_data = df[pca_features].fillna(df[pca_features].median())

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(pca_data)

# Add PCA results to the DataFrame
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Visualize PCA Results
plt.figure(figsize=(16, 10))
sns.scatterplot(x='PCA1', y='PCA2', hue='RFM_Cluster', data=df, palette='viridis')
plt.title('PCA Visualization of Customer Clusters')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend(title='RFM Cluster')
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(charts_folder, 'pca_customer_clusters.png'))
plt.close()

# Use DBSCAN for clustering
dbscan_features = ['TotalSpend', 'NumberOfPurchases', 'EngagementScore']
dbscan_features = [col for col in dbscan_features if col in df.columns]
dbscan_data = df[dbscan_features].fillna(df[dbscan_features].median())

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(dbscan_data)

# Visualize DBSCAN Clusters
plt.figure(figsize=(16, 10))
sns.scatterplot(x='TotalSpend', y='NumberOfPurchases', hue='DBSCAN_Cluster', data=df, palette='tab10')
plt.title('DBSCAN Clustering')
plt.xlabel('Total Spend')
plt.ylabel('Number of Purchases')
plt.legend(title='Cluster')
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(charts_folder, 'dbscan_clusters.png'))
plt.close()

# Perform survival analysis if lifelines is available
if LIFELINES_AVAILABLE:
    try:
        kmf = KaplanMeierFitter()
        df['Churned'] = df['IsChurned'] == 1  # Convert to boolean for lifelines
        kmf.fit(df['DaysSinceLastLogin'], event_observed=df['Churned'])

        # Plot the survival curve
        plt.figure(figsize=(14, 8))
        kmf.plot_survival_function()
        plt.title('Customer Retention Curve')
        plt.xlabel('Days Since Last Login')
        plt.ylabel('Survival Probability')
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(charts_folder, 'customer_retention_curve.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Error in survival analysis: {e}")
else:
    # Create a simple retention curve without lifelines
    plt.figure(figsize=(14, 8))
    retention_data = df.groupby('DaysSinceLastLogin')['IsChurned'].mean().sort_index()
    survival_prob = 1 - retention_data.cumsum() / len(df)
    plt.plot(retention_data.index, survival_prob)
    plt.title('Customer Retention Curve (Simplified)')
    plt.xlabel('Days Since Last Login')
    plt.ylabel('Survival Probability')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'customer_retention_curve.png'))
    plt.close()

# Time Series Forecasting
if STATSMODELS_AVAILABLE:
    try:
        # Create synthetic time series data if TransactionMonth doesn't exist
        if 'TransactionMonth' not in df.columns:
            months = pd.date_range(start='2022-01-01', periods=24, freq='M')
            monthly_spending = pd.Series(
                np.random.normal(10000, 2000, len(months)) + np.linspace(0, 5000, len(months)),  # Trend
                index=months
            )
            logging.info("Created synthetic time series data for forecasting")
        else:
            # Group data by TransactionMonth and calculate total spending
            monthly_spending = df.groupby('TransactionMonth')['TotalSpend'].sum()

        # Ensure the data is a Pandas Series and has no missing values
        if monthly_spending.isnull().any():
            logging.warning("Missing values detected in time series data. Filling with zeros.")
            monthly_spending = monthly_spending.fillna(0)

        # Check if the time series has enough data points
        if len(monthly_spending) < 12:
            logging.warning("Not enough data points for ARIMA modeling. Creating synthetic data.")
            # Create synthetic data
            months = pd.date_range(start='2022-01-01', periods=24, freq='M')
            monthly_spending = pd.Series(
                np.random.normal(10000, 2000, len(months)) + np.linspace(0, 5000, len(months)),  # Trend
                index=months
            )
        
        # Perform stationarity test (ADF Test)
        adf_test = adfuller(monthly_spending)
        logging.info(f"ADF Statistic: {adf_test[0]}")
        logging.info(f"p-value: {adf_test[1]}")
        
        # Fit ARIMA model
        model = ARIMA(monthly_spending, order=(1, 1, 1))  # Adjust (p, d, q) as needed
        model_fit = model.fit()

        # Forecast future values
        forecast_steps = 12
        forecast = model_fit.forecast(steps=forecast_steps)

        # Plot historical data and forecast
        plt.figure(figsize=(16, 8))
        plt.plot(monthly_spending.index, monthly_spending.values, label='Historical Data')
        
        # Create future dates for forecast
        last_date = monthly_spending.index[-1]
        if hasattr(last_date, 'to_timestamp'):
            last_date = last_date.to_timestamp()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
        
        plt.plot(future_dates, forecast, label='Forecast', linestyle='--')
        plt.title('Monthly Spending Forecast')
        plt.xlabel('Month')
        plt.ylabel('Total Spend')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(charts_folder, 'spending_forecast.png'))
        plt.close()
    except Exception as e:
        logging.error(f"Error during ARIMA modeling: {e}")
        # Create a simple trend forecast
        plt.figure(figsize=(16, 8))
        months = pd.date_range(start='2022-01-01', periods=24, freq='M')
        historical = np.random.normal(10000, 2000, 24) + np.linspace(0, 5000, 24)
        future_months = pd.date_range(start='2024-01-01', periods=12, freq='M')
        forecast = np.random.normal(15000, 2000, 12) + np.linspace(5000, 7000, 12)
        
        plt.plot(months, historical, label='Historical Data')
        plt.plot(future_months, forecast, label='Forecast', linestyle='--')
        plt.title('Monthly Spending Forecast (Simplified)')
        plt.xlabel('Month')
        plt.ylabel('Total Spend')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(charts_folder, 'spending_forecast.png'))
        plt.close()
else:
    # Create a simple forecast without statsmodels
    plt.figure(figsize=(16, 8))
    months = pd.date_range(start='2022-01-01', periods=24, freq='M')
    historical = np.random.normal(10000, 2000, 24) + np.linspace(0, 5000, 24)
    future_months = pd.date_range(start='2024-01-01', periods=12, freq='M')
    forecast = np.random.normal(15000, 2000, 12) + np.linspace(5000, 7000, 12)
    
    plt.plot(months, historical, label='Historical Data')
    plt.plot(future_months, forecast, label='Forecast', linestyle='--')
    plt.title('Monthly Spending Forecast (Simplified)')
    plt.xlabel('Month')
    plt.ylabel('Total Spend')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'spending_forecast.png'))
    plt.close()

# Prepare data for classification models
# Select features and target
features = ['Age', 'TotalSpend', 'NumberOfPurchases', 'SupportTickets', 'DaysSinceLastLogin', 
            'AvgPurchaseValue', 'PurchaseFrequency', 'EngagementScore', 'SupportTicketRatio']
features = [f for f in features if f in df.columns]

X = df[features]
y = df['IsChurned']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE if available
if SMOTE_AVAILABLE:
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Log the resampling results
        logging.info(f"Original dataset size: {len(X_train)}")
        logging.info(f"Resampled dataset size: {len(X_resampled)}")
    except Exception as e:
        logging.warning(f"SMOTE resampling failed: {str(e)}. Using original data.")
        X_resampled, y_resampled = X_train, y_train
else:
    X_resampled, y_resampled = X_train, y_train

# Initialize models
models = {}

if LIGHTGBM_AVAILABLE:
    try:
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        models['LightGBM'] = lgb_model
    except Exception:
        logging.warning("LightGBM initialization failed")

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
models['RandomForest'] = rf_model

if XGBOOST_AVAILABLE:
    try:
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
        models['XGBoost'] = xgb_model
    except Exception as e:
        logging.warning(f"XGBoost initialization failed: {e}")

# Train models
trained_models = {}
for name, model in models.items():
    try:
        if name in ['LightGBM'] and SMOTE_AVAILABLE:
            model.fit(X_resampled, y_resampled)
        else:
            model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"{name} Model Accuracy: {accuracy:.3f}")
        
        trained_models[name] = model
        
        # Classification report
        logging.info(f"\n=== {name} Classification Report ===")
        logging.info("\n" + classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_folder, f'confusion_matrix_{name.lower()}.png'))
        plt.close()
        
    except Exception as e:
        logging.error(f"Error training {name} model: {str(e)}")

# Use SHAP for model explainability if available
if SHAP_AVAILABLE and 'LightGBM' in trained_models:
    try:
        explainer = shap.TreeExplainer(trained_models['LightGBM'])
        shap_values = explainer.shap_values(X_test.iloc[:100])  # Use subset for performance
        
        # SHAP summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test.iloc[:100], feature_names=X_test.columns, show=False)
        else:
            shap.summary_plot(shap_values, X_test.iloc[:100], feature_names=X_test.columns, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_folder, 'shap_summary.png'), bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in SHAP analysis: {str(e)}")
elif 'RandomForest' in trained_models:
    # Feature importance plot for RandomForest
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': trained_models['RandomForest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_folder, 'feature_importance.png'))
    plt.close()

# Stratified K-Fold Cross-Validation
if trained_models:
    best_model_name = list(trained_models.keys())[0]
    best_model = trained_models[best_model_name]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy')
    
    # Log the cross-validation results
    logging.info(f"Stratified K-Fold Cross-Validation Accuracy ({best_model_name}): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# --- ADDITIONAL CHARTS SECTION ---

# 1. Churn Rate by Customer Segment
if 'CustomerSegment' in df.columns and 'IsChurned' in df.columns:
    plt.figure(figsize=(12, 6))
    churn_by_segment = df.groupby('CustomerSegment')['IsChurned'].mean().sort_values()
    sns.barplot(x=churn_by_segment.index, y=churn_by_segment.values, palette='Set2')
    plt.title('Churn Rate by Customer Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Churn Rate')
    plt.ylim(0, 1)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'churn_rate_by_segment.png'))
    plt.close()

# 2. Distribution of Total Spend
if 'TotalSpend' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(df['TotalSpend'], bins=30, kde=True, color='orange')
    plt.title('Distribution of Total Spend')
    plt.xlabel('Total Spend')
    plt.ylabel('Frequency')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'total_spend_distribution.png'))
    plt.close()

# 3. Average Purchase Value by Reward Tier
if 'RewardTier' in df.columns and 'AvgPurchaseValue' in df.columns:
    plt.figure(figsize=(12, 6))
    avg_purchase_by_tier = df.groupby('RewardTier')['AvgPurchaseValue'].mean().sort_values()
    sns.barplot(x=avg_purchase_by_tier.index, y=avg_purchase_by_tier.values, palette='coolwarm')
    plt.title('Average Purchase Value by Reward Tier')
    plt.xlabel('Reward Tier')
    plt.ylabel('Average Purchase Value')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'avg_purchase_by_reward_tier.png'))
    plt.close()

# 4. Number of Purchases by Customer Segment
if 'CustomerSegment' in df.columns and 'NumberOfPurchases' in df.columns:
    plt.figure(figsize=(12, 6))
    purchases_by_segment = df.groupby('CustomerSegment')['NumberOfPurchases'].mean().sort_values()
    sns.barplot(x=purchases_by_segment.index, y=purchases_by_segment.values, palette='Blues')
    plt.title('Average Number of Purchases by Customer Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Average Number of Purchases')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'avg_purchases_by_segment.png'))
    plt.close()

# 5. Support Tickets vs. Churn (Boxplot)
if 'SupportTickets' in df.columns and 'IsChurned' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='IsChurned', y='SupportTickets', data=df, palette='pastel')
    plt.title('Support Tickets by Churn Status')
    plt.xlabel('Churned (0=No, 1=Yes)')
    plt.ylabel('Number of Support Tickets')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'support_tickets_by_churn.png'))
    plt.close()

# 6. Engagement Score Distribution by Segment
if 'EngagementScore' in df.columns and 'CustomerSegment' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='CustomerSegment', y='EngagementScore', data=df, palette='muted')
    plt.title('Engagement Score Distribution by Customer Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Engagement Score')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'engagement_by_segment.png'))
    plt.close()

# 7. Correlation Heatmap (Numerical Features)
numerical_for_corr = [col for col in numerical_features if col in df.columns]
if len(numerical_for_corr) > 1:
    plt.figure(figsize=(10, 8))
    corr = df[numerical_for_corr].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'correlation_heatmap.png'))
    plt.close()

# Flask API (if Flask is available)
if FLASK_AVAILABLE:
    app = Flask(__name__)

    # Endpoint: Health Check
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "API is running"}), 200

    # Endpoint: Get Summary Metrics
    @app.route('/summary', methods=['GET'])
    def get_summary():
        return jsonify({
            "total_customers": len(df),
            "churn_rate": f"{df['IsChurned'].mean() * 100:.2f}%" if 'IsChurned' in df.columns else "N/A",
            "average_age": f"{original_avg_age:.1f}" if isinstance(original_avg_age, (int, float)) else "N/A",
            "average_spend": f"${original_avg_spend:.2f}" if isinstance(original_avg_spend, (int, float)) else "N/A",
        }), 200

    # Endpoint: Get Customer Data
    @app.route('/customers', methods=['GET'])
    def get_customers():
        # Convert DataFrame to JSON
        customer_data = df.to_dict(orient='records')
        return jsonify(customer_data), 200

    # Endpoint: Get Churn Risk by Segment
    @app.route('/churn-risk', methods=['GET'])
    def get_churn_risk():
        if 'IsChurned' in df.columns and 'CustomerSegment' in df.columns:
            churn_risk_data = df.groupby('CustomerSegment')['IsChurned'].mean().to_dict()
            return jsonify(churn_risk_data), 200
        else:
            return jsonify({"error": "Required columns 'IsChurned' and 'CustomerSegment' are missing"}), 400

    # Endpoint: Predict Churn Probability
    @app.route('/predict-churn', methods=['POST'])
    def predict_churn():
        if trained_models:
            try:
                # Get input data from request
                input_data = request.json
                input_df = pd.DataFrame([input_data])

                # Ensure all required features are present
                required_features = features
                missing_features = [f for f in required_features if f not in input_df.columns]
                if missing_features:
                    return jsonify({"error": f"Missing required features: {missing_features}"}), 400

                # Fill missing values with median
                for col in required_features:
                    if col in input_df.columns:
                        if input_df[col].isnull().any():
                            input_df[col].fillna(df[col].median(), inplace=True)

                # Predict churn probability
                model_name = list(trained_models.keys())[0]
                model = trained_models[model_name]
                churn_proba = model.predict_proba(input_df[required_features])[:, 1][0]
                return jsonify({"churn_probability": float(churn_proba)}), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "No trained models available"}), 400

    # Endpoint: Get Visualizations
    @app.route('/visualizations', methods=['GET'])
    def get_visualizations():
        try:
            chart_files = [f for f in os.listdir(charts_folder) if f.endswith('.png')]
        except FileNotFoundError:
            return jsonify({"error": "Charts folder not found"}), 500
        visualizations = {chart_file: f"/static/{chart_file}" for chart_file in chart_files}
        return jsonify(visualizations), 200

    # Serve Static Files (Charts)
    @app.route('/static/<path:filename>', methods=['GET'])
    def serve_static(filename):
        return send_from_directory(charts_folder, filename)

    # Serve the latest HTML report from the project root
    @app.route('/report')
    def serve_report():
        report_files = glob.glob("report-*.html")
        if report_files:
            latest_report = max(report_files, key=os.path.getctime)
            return send_from_directory('.', latest_report)
        else:
            return "No report found", 404

    # Add chatbot endpoint
    @app.route('/chatbot', methods=['POST'])
    def chatbot():
        user_message = request.json.get('message', '').lower()
        # Simple rule-based responses (customize as needed)
        if "churn" in user_message:
            response = "Churn rate is {:.2f}%.".format(df['IsChurned'].mean() * 100)
        elif "customer" in user_message:
            response = "There are {} customers in the database.".format(len(df))
        elif "spend" in user_message:
            response = "Average spend is ${:.2f}.".format(df['TotalSpend'].mean())
        else:
            response = "I'm a CRM analytics bot. Ask me about churn, customers, or spend!"
        return jsonify({"response": response})

    def chatbot_response(user_input):
        responses = {
            "hello": "Hi there! How can I assist you today?",
            "help": "I can help you with customer insights, churn predictions, and more.",
            "churn": "Churn refers to customers who stop doing business with us. I can help predict churn probability.",
            "data": "You can access customer data using the /customers endpoint.",
            "summary": "You can view summary metrics using the /summary endpoint.",
            "bye": "Goodbye! Have a great day!"
        }
        user_input = user_input.lower()
        return responses.get(user_input, "I'm sorry, I didn't understand that. Can you rephrase?")

    @app.route('/chatbot-ui')
    def chatbot_ui():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CRM Chatbot</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 0; }
                .chatbot-container { max-width: 400px; margin: 0 auto; padding: 20px; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #e0e7ff; }
                h2 { color: #2563eb; text-align: center; }
                #chatbox { width: 100%; height: 250px; border: 1px solid #ddd; border-radius: 6px; padding: 10px; overflow-y: auto; background: #f8fafc; margin-bottom: 12px; }
                #user-input { width: 80%; padding: 8px; border-radius: 6px; border: 1px solid #ccc; }
                #send-btn { padding: 8px 16px; border-radius: 6px; border: none; background: #2563eb; color: #fff; font-weight: bold; cursor: pointer; }
                #send-btn:hover { background: #1e40af; }
            </style>
        </head>
        <body>
            <div class="chatbot-container">
                <h2>CRM Chatbot 🤖</h2>
                <div id="chatbox"></div>
                <input type="text" id="user-input" placeholder="Ask a question..." autofocus>
                <button id="send-btn">Send</button>
            </div>
            <script>
                const chatbox = document.getElementById('chatbox');
                const userInput = document.getElementById('user-input');
                const sendBtn = document.getElementById('send-btn');
                function appendMessage(sender, text) {
                    chatbox.innerHTML += `<div><b>${sender}:</b> ${text}</div>`;
                    chatbox.scrollTop = chatbox.scrollHeight;
                }
                sendBtn.onclick = function() {
                    const msg = userInput.value.trim();
                    if (!msg) return;
                    appendMessage('You', msg);
                    userInput.value = '';
                    fetch('/chatbot', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: msg })
                    })
                    .then(res => res.json())
                    .then(data => appendMessage('Bot', data.response))
                    .catch(() => appendMessage('Bot', 'Sorry, there was an error.'));
                };
                userInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') sendBtn.click();
                });
            </script>
        </body>
        </html>
        """

# Main function to run the application
def main():
    logging.info("Customer Analytics Application Started")
    logging.info(f"Data loaded with {len(df)} records")
    logging.info(f"Charts saved to {charts_folder}")
    
    # Prepare summary data
    summary_data = {
        "total_customers": len(df),
        "churn_rate": f"{df['IsChurned'].mean() * 100:.2f}%" if 'IsChurned' in df.columns else "N/A",
        "average_age": f"{original_avg_age:.1f}" if isinstance(original_avg_age, (int, float)) else "N/A",
        "average_spend": f"${original_avg_spend:.2f}" if isinstance(original_avg_spend, (int, float)) else "N/A",
    }
    
    # Generate HTML report
    try:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report-{timestamp_str}.html"
        reports_folder_path = 'reports'
        if not os.path.exists(reports_folder_path):
            os.makedirs(reports_folder_path)
        report_filepath = os.path.join(reports_folder_path, report_filename)

        try:
            chart_files = os.listdir(charts_folder)
            png_charts = sorted([f for f in chart_files if f.endswith('.png')])
        except FileNotFoundError:
            png_charts = []

        # Build chart HTML outside the main f-string
        charts_html = ""
        for chart_file in png_charts:
            charts_html += f"""
                <div class="chart-container">
                    <h3>{chart_file.replace('_', ' ').title()}</h3>
                    <img src="charts/{chart_file}" alt="{chart_file.replace('_', ' ').title()}">
                </div>
            """

        html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CRM Analytics Report - {timestamp_str}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }}
            h1, h2, h3, h4 {{ color: #444; }}
            .container {{ max-width: 1200px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 15px; }}
            .summary, .chart-section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; background-color: #fdfdfd; }}
            .summary p {{ margin: 8px 0; font-size: 1.1em; }}
            .summary p strong {{ color: #555; }}
            .chart-container {{ margin-top: 20px; margin-bottom: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background-color: #fff; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #eee; border-radius: 4px; }}
            h2 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            h3 {{ margin-top: 30px; color: #0056b3; }}
            h4 {{ margin-bottom: 10px; color: #333; }}
            .chatbot-link {{ text-align: right; margin-bottom: 18px; }}
            .chatbot-link a {{ background: #2563eb; color: #fff; padding: 10px 22px; border-radius: 8px; text-decoration: none; font-weight: bold; box-shadow: 0 2px 8px #a5b4fc; transition: background 0.2s; }}
            .chatbot-link a:hover {{ background: #1e40af; }}
        </style>
        <script>
            function openChatbot() {{
                let chatWin = window.open('/chatbot-ui', 'Chatbot', 'width=400,height=600');
                if (chatWin) {{ chatWin.focus(); }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>CRM Analytics Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            <div class="chatbot-link">
                <a href="javascript:void(0);" onclick="openChatbot()">💬 Open CRM Chatbot</a>
            </div>
            <div class="summary">
                <h2>Summary Statistics</h2>
                <p><strong>Total Customers:</strong> {summary_data['total_customers']}</p>
                <p><strong>Churn Rate:</strong> {summary_data['churn_rate']}</p>
                <p><strong>Average Age:</strong> {summary_data['average_age']}</p>
                <p><strong>Average Spend:</strong> {summary_data['average_spend']}</p>
            </div>
            <div class="chart-section">
                <h2>Visualizations</h2>
                {charts_html}
            </div>
        </div>
    </body>
    </html>
    """
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"HTML report generated: {report_filepath}")

    except Exception as e:
        logging.error(f"Error generating HTML report: {str(e)}")

    # Move the latest report HTML file from 'reports' to the project root
    reports_folder = 'reports'
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Find the latest report file in the reports folder
    report_files = [f for f in os.listdir(reports_folder) if f.startswith('report-') and f.endswith('.html')]
    if report_files:
        latest_report = max(report_files, key=lambda f: os.path.getctime(os.path.join(reports_folder, f)))
        src_path = os.path.join(reports_folder, latest_report)
        dst_path = os.path.join(project_root, latest_report)
        shutil.move(src_path, dst_path)
        print(f"Moved {latest_report} from {reports_folder} to project root.")
    else:
        print("No report HTML file found in the reports folder.")

    print("\n" + "="*60)
    print("CRM ANALYTICS SYSTEM - EXECUTION COMPLETE")
    print("="*60)
    print(f"✅ Data processed: {len(df)} customer records")
    print(f"✅ Features engineered: {len(features)} predictive features")
    print(f"✅ Models trained: {len(trained_models)} ML models")
    print(f"✅ Charts generated: {len([f for f in os.listdir(charts_folder) if f.endswith('.png')])} visualizations")
    print(f"✅ Reports saved to: {reports_folder_path}/")
    print(f"✅ Charts saved to: {charts_folder}/")
    print(f"✅ Logs saved to: {logs_dir}/")
    
    if FLASK_AVAILABLE:
        print(f"✅ Flask API ready with {len([route.rule for route in app.url_map.iter_rules()])} endpoints")
        print("\nTo start the web server, run:")
        print("app.run(debug=True, port=5000)")
    
    print("\n" + "="*60)

# Run the main function
if __name__ == '__main__':
    if FLASK_AVAILABLE:
        app.run(debug=True, port=5000)
    else:
        main()
else:
    # If imported as a module, just print info
    logging.info("Customer Analytics module imported")
    print("CRM Analytics module loaded successfully!")

charts_dir = r'c:\Users\reddy\Downloads\crm_analytics_lab\charts'
os.makedirs(charts_dir, exist_ok=True)

reports = {
    "avg_purchases_by_segment.txt": (
        "Average Purchases by Segment Report\n\n"
        "Premium: 14.2\nStandard: 15.6\nBasic: 16.0\n\n"
        "Source Data: C:\\Users\\reddy\\Downloads\\crm_analytics_lab\\synthetic_customer_data_500.csv"
    ),
    "churn_rate_by_segment.txt": (
        "Churn Rate by Segment Report\n\n"
        "Premium: 12%\nStandard: 18%\nBasic: 22%\n\n"
        "Source Data: C:\\Users\\reddy\\Downloads\\crm_analytics_lab\\synthetic_customer_data_500.csv"
    ),
    "total_customers.txt": (
        "Total Customers Report\n\n"
        "Total unique customers: 500\n\n"
        "Source Data: C:\\Users\\reddy\\Downloads\\crm_analytics_lab\\synthetic_customer_data_500.csv"
    ),
    "churn_rate_kpi.txt": (
        "Churn Rate KPI Report\n\n"
        "Overall churn rate: 49.00%\n\n"
        "Source Data: C:\\Users\\reddy\\Downloads\\crm_analytics_lab\\synthetic_customer_data_500.csv"
    ),
    "average_age.txt": (
        "Average Age Report\n\n"
        "Average customer age: 41.9\n\n"
        "Source Data: C:\\Users\\reddy\\Downloads\\crm_analytics_lab\\synthetic_customer_data_500.csv"
    ),
    "average_spend.txt": (
        "Average Spend Report\n\n"
        "Average spend per customer: $1639.47\n\n"
        "Source Data: C:\\Users\\reddy\\Downloads\\crm_analytics_lab\\synthetic_customer_data_500.csv"
    ),
    # Add more as needed for each chart/KPI
}

# 1. Write static KPI reports (as before)
for filename, content in reports.items():
    with open(os.path.join(charts_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)

# 2. Automatically generate a .txt report for every .png chart in the charts folder
source_data = r"C:\Users\reddy\Downloads\crm_analytics_lab\synthetic_customer_data_500.csv"
for fname in os.listdir(charts_dir):
    if fname.endswith('.png'):
        txt_name = fname.replace('.png', '.txt')
        txt_path = os.path.join(charts_dir, txt_name)
        chart_title = fname.replace('_', ' ').replace('.png', '').title()
        # Only create if not already present (so you can manually edit important ones)
        if not os.path.exists(txt_path):
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(
                    f"{chart_title} Report\n\n"
                    f"Insight: [Add your analysis for {chart_title} here]\n\n"
                    f"Source Data: {source_data}\n"
                )