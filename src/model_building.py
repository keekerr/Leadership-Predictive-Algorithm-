import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Data Preprocessing Function
def preprocess_data():
    employee_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\employee_data.csv')
    engagement_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\employee_engagement_survey_data.csv')
    recruitment_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\recruitment_data.csv')
    training_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\training_and_development_data.csv')

    # Print the columns of each DataFrame to debug
    print("Employee Data Columns:", employee_data.columns)
    print(employee_data.head())
    print("Engagement Data Columns:", engagement_data.columns)
    print(engagement_data.head())
    print("Recruitment Data Columns:", recruitment_data.columns)
    print(recruitment_data.head())
    print("Training Data Columns:", training_data.columns)
    print(training_data.head())

    # Merging Data Sets
    merged_data = pd.merge(employee_data, engagement_data, on='employee_id', how='inner')
    merged_data = pd.merge(merged_data, recruitment_data, on='employee_id', how='inner')
    merged_data = pd.merge(merged_data, training_data, on='employee_id', how='inner')

    # Missing Values
    merged_data.dropna(inplace=True)

    # Categorical Values (Encode)
    merged_data = pd.get_dummies(merged_data)

    # Ensure the directory exists
    output_dir = '../data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save Data
    merged_data.to_csv(os.path.join(output_dir, 'preprocessed_data.csv'), index=False)

    return merged_data

# Model Building Function
def build_model():
    # Preprocess the data
    data = preprocess_data()
    
    # Features and target variable
    X = data.drop(columns=['leadership_potential'])  # Assuming 'leadership_potential' is the target variable
    y = data['leadership_potential']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Print evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Ensure the directory exists
    output_dir = '../models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model
    joblib.dump(model, os.path.join(output_dir, 'leadership_model.pkl'))

if __name__ == "__main__":
    build_model()