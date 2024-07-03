import pandas as pd
import os

# Uploading Data Sets
def preprocess_data():
    # Adjust the paths as per your project structure
    employee_data_path = r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\employee_data.csv'
    engagement_data_path = r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\employee_engagement_survey_data.csv'
    recruitment_data_path = r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\recruitment_data.csv'
    training_data_path = r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\training_and_development_data.csv'
    
    employee_data = pd.read_csv(employee_data_path)
    engagement_data = pd.read_csv(engagement_data_path)
    recruitment_data = pd.read_csv(recruitment_data_path)
    training_data = pd.read_csv(training_data_path)

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
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save Data
    preprocessed_data_path = os.path.join(output_dir, 'preprocessed_data.csv')
    merged_data.to_csv(preprocessed_data_path, index=False)

    print(f"Preprocessed data saved to {preprocessed_data_path}")

    return merged_data

if __name__ == "__main__":
    preprocess_data()
