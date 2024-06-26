# Imports
import pandas as pd

# Uploading Data Sets
def preprocess_data():
    employee_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\employee_data.csv')
    engagement_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\employee_engagement_survey_data.csv')
    recruitment_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\recruitment_data.csv')
    training_data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\MIS581 Port folio Data Sets\training_and_development_data.csv')

    # Merging Data Sets
    merged_data = pd.merge(employee_data, engagement_data, on='employee_id', how='inner')
    merged_data = pd.merge(merged_data, recruitment_data, on='employee_id', how='inner')
    merged_data = pd.merge(merged_data, training_data, on='employee_id', how='inner')

    # Missing Values
    merged_data.dropna(inplace=True)

    # Categorical Values (Encode)
    merged_data = pd.get_dummies(merged_data)

    # Save Data
    merged_data.to_csv('../data/preprocessed_data.csv', index=False)

    return merged_data

if __name__ == "__main__":
    preprocess_data()