# src/data_preprocessing.py

import pandas as pd
import os

def load_and_prepare_data(data_path='data'):
    """
    Load and merge the Instacart dataset files into a single DataFrame.
    
    Args:
        data_path (str): Path to the data directory.
        
    Returns:
        pd.DataFrame: Merged DataFrame of user-product interactions.
    """
    print("🔄 Loading data...")

    # Load datasets
    orders = pd.read_csv(os.path.join(data_path, 'orders.csv'))
    order_products_train = pd.read_csv(os.path.join(data_path, 'order_products_train.csv'))
    order_products_prior = pd.read_csv(os.path.join(data_path, 'order_products_prior.csv'))
    products = pd.read_csv(os.path.join(data_path, 'products.csv'))
    aisles = pd.read_csv(os.path.join(data_path, 'aisles.csv'))
    departments = pd.read_csv(os.path.join(data_path, 'departments.csv'))

    print("✅ Data loaded.")
    
    # Merge the data
    print("🔄 Merging data...")
    # Merging order products train with orders to get user-level information
    train_merged = order_products_train.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
    train_merged = train_merged.merge(products[['product_id', 'aisle_id', 'department_id']], on='product_id', how='left')
    print(f"✅ Merged train data. Columns: {train_merged.columns.tolist()}")

    # Similarly, merge order_products_prior with orders to get prior purchase data (for feature engineering)
    prior_merged = order_products_prior.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
    prior_merged = prior_merged.merge(products[['product_id', 'aisle_id', 'department_id']], on='product_id', how='left')
    print(f"✅ Merged prior data. Columns: {prior_merged.columns.tolist()}")
    
    # Return both merged datasets
    return train_merged, prior_merged
