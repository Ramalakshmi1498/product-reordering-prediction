import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from src.data_preprocessing import load_and_prepare_data, generate_features

def train_model():
    # Load and preprocess the data
    print("🔄 Loading and preparing data...")
    df = load_and_prepare_data('data')
    features_df = generate_features(df)

    # Prepare features and labels
    X = features_df.drop(['user_id', 'product_id', 'order_id', 'reordered'], axis=1)  # Features
    y = features_df['reordered']  # Label (target)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model (Random Forest for simplicity)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("🔍 Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'model.pkl')
    print("✅ Model saved as 'model.pkl'.")

if __name__ == "__main__":
    train_model()
