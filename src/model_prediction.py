import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

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

# Model Prediction Function
def predict_model():
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

    # Load the model
    model_path = '../models/leadership_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    # Predict and evaluate
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

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    predict_model()
