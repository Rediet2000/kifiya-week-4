import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import logging
# Define paths to the dataset


# Setup logger for logging tasks
logging.basicConfig(filename='./data/eda_task.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger()

# 1. Load Data
def load_data(train_path, test_path, store_path):
    logger.info("Loading data...")
    dtype_spec = {
        # Example: 'ColumnName': str,
    }
    try:
        train = pd.read_csv(train_path, dtype=dtype_spec, low_memory=False)
        test = pd.read_csv(test_path, dtype=dtype_spec, low_memory=False)
        store = pd.read_csv(store_path, dtype=dtype_spec, low_memory=False)
        logger.info(f"Train data shape: {train.shape}")
        logger.info(f"Test data shape: {test.shape}")
        logger.info(f"Test data shape: {store.shape}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise e  # Re-raise the error to stop execution
    return train, test, store

# 2. Data Cleaning Function
def clean_data(df):
    logger.info("Cleaning data - Handling missing values...")
    missing_cols = df.columns[df.isnull().any()]
    logger.info(f"Missing columns: {missing_cols}")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    logger.info("Missing values handled.")
    
    # Handling outliers using Isolation Forest
    logger.info("Detecting outliers...")
    iso_forest = IsolationForest(contamination=0.01)
    outliers = iso_forest.fit_predict(df[num_cols])
    df = df[outliers != -1]  # Remove outliers
    logger.info(f"Outliers removed: {sum(outliers == -1)}")
    
    return df

# 3. Feature Engineering
def feature_engineering(df):
    logger.info("Performing feature engineering...")
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert to datetime and coerce errors
        if df['Date'].isnull().any():
            logger.warning("Some dates could not be converted and have been set to NaT.")
            df = df.dropna(subset=['Date'])  # Drop rows where Date conversion failed
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
        
        # Example: Categorizing holidays and promotions
        df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x in ['a', 'b', 'c'] else 0)
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
    
    logger.info("Feature engineering completed.")
    
    return df

# 4. Exploratory Data Analysis (EDA) with Visualizations
def exploratory_analysis(train, test, store):
    logger.info("Starting EDA...")

    # Check distribution of promotions in train vs test datasets
    logger.info("Analyzing promotions distribution between train and test sets...")
    plt.figure()
    sns.histplot(train['Promo'], label='Train', kde=False, color='blue', bins=2)
    sns.histplot(test['Promo'], label='Test', kde=False, color='orange', bins=2)
    plt.title('Promo Distribution in Train vs Test')
    plt.legend()
    plt.savefig('promo_distribution.png')  # Save the figure
    plt.close()

    # Correlation analysis between sales, customers, promos
    logger.info("Correlation analysis...")
    plt.figure()
    corr_matrix = train[['Sales', 'Customers', 'Promo']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Sales, Customers, and Promo')
    plt.savefig('correlation_matrix.png')  # Save the figure
    plt.close()

    # Example: Visualize sales before, during, and after holidays
    logger.info("Analyzing sales behavior around holidays...")
    plt.figure()
    holidays_sales = train.groupby(['IsHoliday', 'Promo'])['Sales'].mean().unstack()
    holidays_sales.plot(kind='bar', stacked=True)
    plt.title('Sales during Holidays and Promo Periods')
    plt.savefig('sales_holidays.png')  # Save the figure
    plt.close()

    logger.info("EDA completed.")

# Main function to run EDA
def run_eda(train_path, test_path, store_paths):
    logger.info("EDA task started.")
    train, test, store = load_data(train_path, test_path, store_paths)
    
    # Clean the data
    train_cleaned = clean_data(train)
    test_cleaned = clean_data(test)
    store_cleaned = clean_data(store)
    
    # Feature engineering
    train_fe = feature_engineering(train_cleaned)
    test_fe = feature_engineering(test_cleaned)
    store_fe = feature_engineering(store_cleaned)
    
    # Run exploratory data analysis
    exploratory_analysis(train_fe, test_fe, store_fe)
    
    logger.info("EDA task completed.")

run_eda('./rossmann-store-sales/train.csv', './rossmann-store-sales/test.csv', './rossmann-store-sales/store.csv')
