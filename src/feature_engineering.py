import pandas as pd
from sklearn.model_selection import train_test_split

def generate_features(train_merged, prior_merged):
    print("🔄 Generating features...")

    # Combine train and prior data to create a full user-product interaction dataset
    features = pd.concat([train_merged, prior_merged], axis=0, ignore_index=True)
    print("Columns in combined dataset:", features.columns)

    # Create features based on user-product grouping
    features['total_orders'] = features.groupby(['user_id', 'product_id'])['order_id'].transform('count')
    features['total_reorders'] = features.groupby(['user_id', 'product_id'])['reordered'].transform('sum')

    # For simplicity, let's keep other columns, then drop only after feature engineering
    # Now, prepare features and labels from the train_merged subset only
    train_features = features[features['order_id'].isin(train_merged['order_id'])]

    # Label (target)
    y = train_features['reordered']

    # Drop columns that are not features (drop only after feature engineering)
    X = train_features.drop(columns=['user_id', 'product_id', 'order_id', 'reordered', 'add_to_cart_order', 'aisle_id', 'department_id'])

    # Split train/validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("✅ Features generated successfully.")

    return X_train, X_val, y_train, y_val
