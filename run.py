# run.py

from src.data_preprocessing import load_and_prepare_data
from src.feature_engineering import generate_features
from src.model import build_model
from src.evaluate import evaluate_model

def main():
    print("🚀 Starting Deep Learning-Based Reorder Prediction")
    
    # 1. Load and merge data
    train_merged, prior_merged = load_and_prepare_data()
    
    # 2. Generate features
    X_train, X_val, y_train, y_val = generate_features(train_merged, prior_merged)
    
    # 3. Build model
    model = build_model(X_train.shape[1:])
    
    # 4. Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
    
    # 5. Evaluate model
    evaluate_model(model, X_val, y_val)
    
    print("🎉 Project complete. Check the output folder for results.")
    
if __name__ == "__main__":
    main()
