import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from lifelines import KaplanMeierFitter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from textblob import TextBlob
import shap
from flask import Flask, jsonify, request, send_from_directory
import importlib.util
import matplotlib

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
sns.set(style="whitegrid")

data_path = r'C:\Users\reddy\Downloads\crm_analytics_lab\synthetic_customer_data_500.csv'
logging.info(f"Attempting to load data from file: {data_path}")

try:
    if not os.path.exists(data_path):
        logging.warning(f"File not found at {data_path}. Creating synthetic data for demonstration.")
        np.random.seed(42)
        num_samples = 500
        customer_ids = [f'CUST{i:05d}' for i in range(1, num_samples + 1)]
        ages = np.random.normal(35, 10, num_samples).astype(int)
        ages = np.clip(ages, 18, 80)
        total_spend = np.random.exponential(500, num_samples)
        num_purchases = np.random.poisson(5, num_samples)
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
            'Latitude': np.random.uniform(*config['default_latitude_range'], size=num_samples),
            'Longitude': np.random.uniform(*config['default_longitude_range'], size=num_samples)
        })
        df.to_csv(data_path, index=False)
        logging.info(f"Synthetic data created and saved to {data_path}")
    else:
        df = pd.read_csv(data_path)
        logging.info("Data loaded successfully!")
    if 'Feedback' not in df.columns:
        df['Feedback'] = np.random.choice(['Great service!', 'Not satisfied', 'Average experience', 
                                         'Excellent support', 'Could be better'], size=len(df))
    if 'Latitude' not in df.columns:
        df['Latitude'] = np.random.uniform(*config['default_latitude_range'], size=len(df))
    if 'Longitude' not in df.columns:
        df['Longitude'] = np.random.uniform(*config['default_longitude_range'], size=len(df))
except Exception as e:
    logging.error(f"Error loading data: {str(e)}")
    df = pd.DataFrame({
        'CustomerID': [f'CUST{i:05d}' for i in range(1, 101)],
        'Age': np.random.normal(35, 10, 100).astype(int),
        'TotalSpend': np.random.exponential(500, 100),
        'NumberOfPurchases': np.random.poisson(5, 100),
        'SupportTickets': np.random.poisson(2, 100),
        'CustomerSegment': np.random.choice(['Premium', 'Standard', 'Basic'], 100),
        'IsChurned': np.random.binomial(1, 0.2, 100),
        'Feedback': np.random.choice(['Great service!', 'Not satisfied'], 100),
        'Latitude': np.random.uniform(*config['default_latitude_range'], size=100),
        'Longitude': np.random.uniform(*config['default_longitude_range'], size=100)
    })
    logging.info("Created fallback data due to loading error")

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

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'TotalSpend', 'NumberOfPurchases', 'SupportTickets', 'DaysSinceLastLogin', 
                      'AvgPurchaseValue', 'PurchaseFrequency', 'EngagementScore', 'SupportTicketRatio']
numerical_features = [col for col in numerical_features if col in df.columns]

# Create a copy to avoid SettingWithCopyWarning
scaled_features = scaler.fit_transform(df[numerical_features])
df[numerical_features] = scaled_features

# Log the scaling process
logging.info("Numerical features have been standardized.")

# Use IsolationForest for anomaly detection
isolation_forest = IsolationForest(random_state=42, contamination=0.05)
df['Anomaly'] = isolation_forest.fit_predict(df[['TotalSpend', 'NumberOfPurchases']])

# Mark anomalies
df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Outlier' if x == -1 else 'Normal')

# Log the number of anomalies detected
logging.info(f"Number of anomalies detected: {df['Anomaly'].value_counts().get('Outlier', 0)}")

# Perform sentiment analysis on customer feedback
# Add a Sentiment column based on the polarity of the feedback
df['Sentiment'] = df['Feedback'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

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
            result = pd.qcut(clean_series, q=unique_values, labels=adjusted_labels)
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

# Apply RFM segmentation
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
kmeans_rfm = KMeans(n_clusters=4, random_state=42)
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

# Perform survival analysis
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

# Time Series Forecasting
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
    plt.plot(monthly_spending, label='Historical Data')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title('Monthly Spending Forecast')
    plt.xlabel('Month')
    plt.ylabel('Total Spend')
    plt.legend()
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'spending_forecast.png'))
    plt.close()
except Exception as e:
    logging.error(f"Error during ARIMA modeling: {e}")

# Prepare data for classification models
# Select features and target
features = ['Age', 'TotalSpend', 'NumberOfPurchases', 'SupportTickets', 'DaysSinceLastLogin', 
            'AvgPurchaseValue', 'PurchaseFrequency', 'EngagementScore', 'SupportTicketRatio']
features = [f for f in features if f in df.columns]

X = df[features]
y = df['IsChurned']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
try:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Log the resampling results
    logging.info(f"Original dataset size: {len(X_train)}")
    logging.info(f"Resampled dataset size: {len(X_resampled)}")
except Exception as e:
    logging.warning(f"SMOTE resampling failed: {str(e)}. Using original data.")
    X_resampled, y_resampled = X_train, y_train

# Initialize models
lgb_model = lgb.LGBMClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Train models
try:
    lgb_model.fit(X_resampled, y_resampled)
    rf_model.fit(X_resampled, y_resampled)
    xgb_model.fit(X_train, y_train)  # XGBoost trained on original data for comparison
    
    # Evaluate the XGBoost model
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    logging.info(f"XGBoost Model Accuracy: {xgb_accuracy:.2f}")
    
    # Add models to the models dictionary
    models = {
        'LightGBM': lgb_model,
        'RandomForest': rf_model,
        'XGBoost': xgb_model
    }
    
    # Use SHAP for model explainability
    explainer = shap.TreeExplainer(models['LightGBM'])
    shap_values = explainer.shap_values(X_test)
    
    # Visualize feature importance using SHAP
    plt.figure(figsize=(20, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'shap_summary.png'))
    plt.close()

    # SHAP force plot (improved spacing)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_for_plot = shap_values[1]
        expected_value_for_plot = explainer.expected_value[1]
    else:
        shap_values_for_plot = shap_values
        expected_value_for_plot = explainer.expected_value
    
    # Create individual force plots for top features
    sample_idx = 0  # Use the first test sample
    
    # Get feature importance for this sample
    feature_importance = np.abs(shap_values_for_plot[sample_idx])
    feature_names = X_test.columns
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = [shap_values_for_plot[sample_idx, i] for i in sorted_idx]
    
    # Create a horizontal bar chart instead of force plot
    plt.figure(figsize=(24, 10))
    colors = ['red' if x < 0 else 'blue' for x in sorted_values]
    y_pos = np.arange(len(sorted_features))
    
    # Create the horizontal bar chart
    bars = plt.barh(y_pos, sorted_values, color=colors)
    
    # Add feature values to the right of each bar
    for i, bar in enumerate(bars):
        feature_name = sorted_features[i]
        feature_value = X_test.iloc[sample_idx][feature_name]
        plt.text(
            bar.get_width() + (0.01 * np.max(np.abs(sorted_values))),
            bar.get_y() + bar.get_height()/2,
            f"{feature_name} = {feature_value:.2f}",
            va='center',
            fontsize=10
        )
    
    # Add base value line
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add expected value
    plt.text(
        0,
        len(sorted_features) + 0.5,
        f"Base value: {expected_value_for_plot:.3f}",
        ha='center',
        fontsize=12,
        fontweight='bold'
    )
    
    # Set labels and title
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.title('Feature Impact on Churn Prediction')
    plt.yticks([])  # Hide y-axis labels since we added them as text
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Increases prediction'),
        Patch(facecolor='red', label='Decreases prediction')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'shap_feature_impact.png'))
    plt.close()
    
    # Create a more readable version of the force plot
    plt.figure(figsize=(28, 6))
    
    # Create a horizontal line for the base value
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Starting point is the base value
    cumulative = 0
    
    # Plot each feature contribution as a segment
    for i, (feature, value) in enumerate(zip(sorted_features, sorted_values)):
        # Plot the segment
        plt.plot([i, i+1], [cumulative, cumulative + value], 
                 color='blue' if value > 0 else 'red', 
                 linewidth=2.5)
        
        # Add the feature name and value
        feature_value = X_test.iloc[sample_idx][feature]
        if value > 0:
            plt.text(i + 0.5, cumulative + value/2, 
                     f"{feature}\n({feature_value:.2f})", 
                     ha='center', va='bottom', 
                     fontsize=9, rotation=90)
        else:
            plt.text(i + 0.5, cumulative + value/2, 
                     f"{feature}\n({feature_value:.2f})", 
                     ha='center', va='top', 
                     fontsize=9, rotation=90)
        
        # Update cumulative value
        cumulative += value
    
    # Add final prediction point
    plt.scatter([len(sorted_features)], [cumulative], color='black', s=50)
    plt.text(len(sorted_features), cumulative, 
             f"Final prediction\n{cumulative+expected_value_for_plot:.3f}", 
             ha='right', va='center', fontsize=10)
    
    # Add base value
    plt.scatter([0], [0], color='black', s=50)
    plt.text(0, 0, f"Base value\n{expected_value_for_plot:.3f}", 
             ha='left', va='center', fontsize=10)
    
    # Set labels and title
    plt.xlabel('Features')
    plt.ylabel('Contribution to prediction')
    plt.title('SHAP Force Plot: Feature Contributions to Churn Prediction')
    
    # Remove x-ticks
    plt.xticks([])
    
    # Add grid for y-axis
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'shap_force_plot_improved.png'))
    plt.close()

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lgb_model, X, y, cv=skf, scoring='accuracy')
    
    # Log the cross-validation results
    logging.info(f"Stratified K-Fold Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    # Train RandomForestClassifier model
    rf_cv_scores = cross_val_score(rf_model, X, y, cv=skf, scoring='accuracy')
    
    # Log the cross-validation results
    logging.info(f"Random Forest Cross-Validation Accuracy: {rf_cv_scores.mean():.2f} ± {rf_cv_scores.std():.2f}")
    
    # Evaluate the LightGBM model
    y_pred_lgb = lgb_model.predict(X_test)
    logging.info("\n=== LightGBM Classification Report ===")
    logging.info("\n" + classification_report(y_test, y_pred_lgb))
    
    # Evaluate the RandomForest model
    y_pred_rf = rf_model.predict(X_test)
    logging.info("\n=== RandomForest Classification Report ===")
    logging.info("\n" + classification_report(y_test, y_pred_rf))
    
    # Confusion Matrix for LightGBM
    cm_lgb = confusion_matrix(y_test, y_pred_lgb)
    disp_lgb = ConfusionMatrixDisplay(confusion_matrix=cm_lgb, display_labels=['Not Churned', 'Churned'])
    fig_lgb, ax_lgb = plt.subplots(figsize=(12, 10))
    disp_lgb.plot(cmap='Blues', ax=ax_lgb)
    plt.title("Confusion Matrix - LightGBM")
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'confusion_matrix_lightgbm.png'))
    plt.close()
    
    # Confusion Matrix for RandomForest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Not Churned', 'Churned'])
    fig_rf, ax_rf = plt.subplots(figsize=(12, 10))
    disp_rf.plot(cmap='Blues', ax=ax_rf)
    plt.title("Confusion Matrix - RandomForest")
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(charts_folder, 'confusion_matrix_randomforest.png'))
    plt.close()
except Exception as e:
    logging.error(f"Error during model training and evaluation: {str(e)}")
    models = {}

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

# Flask API
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
        "average_age": f"{df['Age'].mean():.1f}" if 'Age' in df.columns else "N/A",
        "average_spend": f"${df['TotalSpend'].mean():.2f}" if 'TotalSpend' in df.columns else "N/A",
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
    if 'LightGBM' in models:
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
            churn_proba = models['LightGBM'].predict_proba(input_df[required_features])[:, 1][0]
            return jsonify({"churn_probability": float(churn_proba)}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "LightGBM model is not available"}), 400

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

# Add this chatbot endpoint if not already present
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        response = chatbot_response(user_input)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

# --- Enhanced Dashboard Template with Chatbot Link ---
dashboard_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CRM Analytics Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%); margin: 0; padding: 0; }
        .dashboard-container { max-width: 1500px; margin: 30px auto; background: #fff; border-radius: 18px; box-shadow: 0 4px 24px rgba(0,0,0,0.10); padding: 36px 32px; }
        .kpi-row { display: flex; gap: 32px; margin-bottom: 44px; }
        .kpi-card { flex: 1; background: linear-gradient(120deg, #e0e7ff 0%, #f8fafc 100%); border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); padding: 28px 18px; text-align: center; border: 2px solid #e0e7ff; }
        .kpi-title { font-size: 1.15em; color: #6b7280; margin-bottom: 10px; letter-spacing: 0.5px; }
        .kpi-value { font-size: 2.5em; font-weight: bold; color: #2563eb; text-shadow: 0 2px 8px #e0e7ff; }
        .charts-section { margin-top: 30px; }
        .slider-container { margin-bottom: 36px; }
        .slider-label { font-size: 1.1em; color: #374151; margin-bottom: 8px; }
        .slider-value { font-weight: bold; color: #2563eb; margin-left: 10px; }
        .charts-slider { width: 100%; }
        .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 32px; }
        .chart-card { background: #f3f4f6; border-radius: 12px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); padding: 18px; margin-bottom: 10px; border: 2px solid #e0e7ff; }
        .chart-title { font-size: 1.13em; color: #1e293b; margin-bottom: 12px; font-weight: 500; }
        img { width: 100%; max-width: 100%; border-radius: 8px; border: 1.5px solid #c7d2fe; background: #fff; }
        .chatbot-link { display: block; text-align: right; margin-bottom: 18px; }
        .chatbot-link a { background: #2563eb; color: #fff; padding: 10px 22px; border-radius: 8px; text-decoration: none; font-weight: bold; box-shadow: 0 2px 8px #a5b4fc; transition: background 0.2s; }
        .chatbot-link a:hover { background: #1e40af; }
        @media (max-width: 900px) {
            .kpi-row { flex-direction: column; gap: 18px; }
            .charts-grid { grid-template-columns: 1fr; }
            .chatbot-link { text-align: center; }
        }
        .slider {
            -webkit-appearance: none;
            width: 100%;
            height: 10px;
            border-radius: 5px;
            background: linear-gradient(90deg, #6366f1 0%, #60a5fa 100%);
            outline: none;
            opacity: 0.9;
            transition: opacity .2s;
            margin-bottom: 10px;
        }
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #2563eb;
            cursor: pointer;
            box-shadow: 0 2px 8px #a5b4fc;
            border: 3px solid #fff;
        }
        .slider::-moz-range-thumb {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #2563eb;
            cursor: pointer;
            box-shadow: 0 2px 8px #a5b4fc;
            border: 3px solid #fff;
        }
    </style>
    <script>
        function updateSlider(val) {
            document.getElementById('slider-value').innerText = val;
            let chartCards = document.getElementsByClassName('chart-card');
            for (let i = 0; i < chartCards.length; i++) {
                chartCards[i].style.display = (i < val) ? 'block' : 'none';
            }
        }
        window.onload = function() {
            let slider = document.getElementById('chart-slider');
            updateSlider(slider.value);
        }
        // Chatbot popup logic
        function openChatbot() {
            let chatWin = window.open('/chatbot-ui', 'Chatbot', 'width=400,height=600');
            if (chatWin) { chatWin.focus(); }
        }
    </script>
</head>
<body>
    <div class="dashboard-container">
        <div class="chatbot-link">
            <a href="javascript:void(0);" onclick="openChatbot()">💬 Open CRM Chatbot</a>
        </div>
        <h1 style="text-align:center; color:#2563eb; margin-bottom:36px; letter-spacing:1px;">CRM Analytics Dashboard</h1>
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">Total Customers</div>
                <div class="kpi-value">{{ total_customers }}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Churn Rate</div>
                <div class="kpi-value">{{ churn_rate }}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Average Age</div>
                <div class="kpi-value">{{ average_age }}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Average Spend</div>
                <div class="kpi-value">{{ average_spend }}</div>
            </div>
        </div>
        <div class="charts-section">
            <div class="slider-container">
                <label class="slider-label" for="chart-slider">Charts to Display: 
                    <span class="slider-value" id="slider-value">{{ charts|length }}</span>
                </label>
                <input type="range" min="1" max="{{ charts|length }}" value="{{ charts|length }}" class="slider" id="chart-slider" oninput="updateSlider(this.value)">
            </div>
            <div class="charts-grid">
                {% for chart in charts %}
                <div class="chart-card">
                    <div class="chart-title">{{ chart.title }}</div>
                    <img src="{{ chart.path }}" alt="{{ chart.title }}">
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
    <script>
        document.getElementById('chart-slider').addEventListener('input', function() {
            updateSlider(this.value);
        });
    </script>
</body>
</html>
"""

# Simple chatbot UI page
@app.route('/chatbot-ui')
def chatbot_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CRM Chatbot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; margin: 0; padding: 0; }
            .chat-container { max-width: 400px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 20px; margin-top: 20px; }
            .chat-title { text-align: center; color: #2563eb; font-size: 1.3em; margin-bottom: 18px; }
            .chat-history { height: 320px; overflow-y: auto; border: 1px solid #e0e7ff; border-radius: 8px; padding: 12px; background: #f3f4f6; margin-bottom: 14px; }
            .chat-message { margin-bottom: 10px; }
            .chat-message.user { text-align: right; color: #2563eb; }
            .chat-message.bot { text-align: left; color: #111; }
            .chat-input-row { display: flex; gap: 8px; }
            .chat-input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #c7d2fe; }
            .chat-send { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 10px 18px; font-weight: bold; cursor: pointer; }
            .chat-send:hover { background: #1e40af; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-title">CRM Chatbot</div>
            <div class="chat-history" id="chat-history"></div>
            <div class="chat-input-row">
                <input type="text" id="chat-input" class="chat-input" placeholder="Type your message..." autocomplete="off" />
                <button class="chat-send" onclick="sendMessage()">Send</button>
            </div>
        </div>
        <script>
            function appendMessage(sender, text) {
                var chatHistory = document.getElementById('chat-history');
                var msgDiv = document.createElement('div');
                msgDiv.className = 'chat-message ' + sender;
                msgDiv.innerText = text;
                chatHistory.appendChild(msgDiv);
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
            function sendMessage() {
                var input = document.getElementById('chat-input');
                var message = input.value.trim();
                if (!message) return;
                appendMessage('user', message);
                input.value = '';
                fetch('/chatbot', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.response) {
                        appendMessage('bot', data.response);
                    } else if (data.error) {
                        appendMessage('bot', 'Error: ' + data.error);
                    }
                });
            }
            document.getElementById('chat-input').addEventListener('keydown', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """

# Main function to run the application
def main(original_avg_age_arg, original_avg_spend_arg):
    logging.info("Customer Analytics Application Started")
    logging.info(f"Data loaded with {len(df)} records")
    logging.info(f"Charts saved to {charts_folder}")
    
    # Prepare summary data using passed original values
    summary_data = {
        "total_customers": len(df),  # df should be accessible from global scope here
        "churn_rate": f"{df['IsChurned'].mean() * 100:.2f}%" if 'IsChurned' in df.columns else "N/A",
        "average_age": f"{original_avg_age_arg:.1f}" if isinstance(original_avg_age_arg, (int, float)) else "N/A",
        "average_spend": f"${original_avg_spend_arg:.2f}" if isinstance(original_avg_spend_arg, (int, float)) else "N/A",
    }
    
    # Generate HTML report
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report-{timestamp}.html"
        report_filepath = os.path.join(charts_folder, report_filename)

        chart_files = os.listdir(charts_folder)
        png_charts = sorted([f for f in chart_files if f.endswith('.png')])
        html_charts = sorted([f for f in chart_files if f.endswith('.html') and not f.startswith('report-')])  # Exclude previous reports

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRM Analytics Report - {timestamp}</title>
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
        iframe {{ width: 100%; height: 500px; border: 1px solid #eee; border-radius: 4px; }}
        a {{ color: #007bff; text-decoration: none; font-weight: bold; }}
        a:hover {{ text-decoration: underline; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h3 {{ margin-top: 30px; color: #0056b3; }}
        h4 {{ margin-bottom: 10px; color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CRM Analytics Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
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
            <!-- Add this inside your HTML where you want to display the chart -->

<div class="chart-container">
    <h4>Distribution of Total Spend</h4>
    <img src="/static/total_spend_distribution.png" alt="Distribution of Total Spend">
</div>

"""
        # Add PNG charts
        if png_charts:
            # Mapping of file names to display names
            chart_name_map = {
                'churn_rate_by_segment.png': 'Churn Rate by Customer Segment',
                'total_spend_distribution.png': 'Distribution of Total Spend',
                'avg_purchase_by_reward_tier.png': 'Average Purchase Value by Reward Tier',
                'avg_purchases_by_segment.png': 'Average Number of Purchases by Customer Segment',
                'support_tickets_by_churn.png': 'Support Tickets by Churn Status',
                'engagement_by_segment.png': 'Engagement Score Distribution by Segment',
                'correlation_heatmap.png': 'Correlation Heatmap',
                'customer_sentiment.png': 'Customer Sentiment Distribution',
                'pca_customer_clusters.png': 'PCA Visualization of Customer Clusters',
                'dbscan_clusters.png': 'DBSCAN Clustering',
                'customer_retention_curve.png': 'Customer Retention Curve',
                'spending_forecast.png': 'Monthly Spending Forecast',
                'shap_summary.png': 'SHAP Summary Plot',
                'shap_feature_impact.png': 'SHAP Feature Impact',
                'shap_force_plot_improved.png': 'SHAP Force Plot',
                'confusion_matrix_lightgbm.png': 'Confusion Matrices for LightGBM and RandomForest',
                'confusion_matrix_randomforest.png': 'Confusion Matrices for LightGBM and RandomForest'
            }
            html_content += "<h3>Image Charts</h3>"
            for chart in png_charts:
                chart_path = f"./{chart}"
                chart_title = chart_name_map.get(chart, chart.replace('.png', '').replace('_', ' ').title())
                html_content += f"""
            <div class="chart-container">
                <h4>{chart_title}</h4>
                <img src="{chart_path}" alt="{chart_title}">
            </div>
"""
        # Add HTML charts (as links)
        if html_charts:
            html_content += "<h3>Interactive Charts</h3>"
            for chart in html_charts:
                # Relative path for direct file opening
                chart_path = f"./{chart}"  # Ensure ./ prefix for local file opening
                chart_title = chart.replace('.html', '').replace('_', ' ').title()
                html_content += f"""
            <div class="chart-container">
                <h4>{chart_title}</h4>
                <p><a href="{chart_path}" target="_blank">Open Interactive: {chart_title}</a></p>
            </div>
"""
        html_content += """
        </div>
    </div>
</body>
</html>
"""
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"HTML report generated: {report_filepath}")
        # Optionally open the report in a browser if desired (might require `webbrowser` import)
        # import webbrowser
        # webbrowser.open(f'file://{os.path.realpath(report_filepath)}')

    except Exception as e:
        logging.error(f"Error generating HTML report: {str(e)}")

    # Run Flask app if this is the main module
    if __name__ == '__main__':
        try:
            port = int(os.environ.get("PORT", 5000))
            # Set use_reloader=False if the auto-restarting is bothersome after report generation
            app.run(debug=True, port=port, use_reloader=False)
        except Exception as e:
            logging.error(f"Error starting Flask app: {str(e)}")

# Run the main function
if __name__ == '__main__':
    # Pass the globally calculated original stats to main
    main(original_avg_age, original_avg_spend) 
else:
    # If imported as a module, just print info
    logging.info("Customer Analytics module imported")