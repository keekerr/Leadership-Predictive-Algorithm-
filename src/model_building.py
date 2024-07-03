import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Data Preprocessing Function
def preprocess_data():
   
    preprocessed_data_path = r'data/preprocessed_data.csv'
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(f"Preprocessed data file not found at {preprocessed_data_path}")
    
    data = pd.read_csv(preprocessed_data_path)
    return data

# Model Building Function
def build_model():
    
    data = preprocess_data()
    
    # Features and target variable
    X = data.drop(columns=['Current Employee Rating'])  # Assuming 'Current Employee Rating' is the target variable
    y = data['Current Employee Rating']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    output_dir = 'models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model
    model_path = os.path.join(output_dir, 'leadership_model.pkl')
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    build_model()
