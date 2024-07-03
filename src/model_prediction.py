import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Data Preprocessing Function
def preprocess_data():
    preprocessed_data_path = r'data/preprocessed_data.csv'
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(f"Preprocessed data file not found at {preprocessed_data_path}")
    
    data = pd.read_csv(preprocessed_data_path)
    return data

# Model Prediction Function
def predict_model():
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

    # Load the model
    model_path = 'models/leadership_model.pkl'
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