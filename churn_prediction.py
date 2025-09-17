import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def preprocess_data(df):
    """Cleans and prepares the data for model training."""
    # Drop Customer ID
    df = df.drop('customerID', axis=1)

    # Convert 'TotalCharges' to numeric, handling missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encode categorical features
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

    return df

def train_model(df):
    """Trains and evaluates a churn prediction model."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("--- Churn Prediction Model Report ---")
    print(classification_report(y_test, predictions))
    print("-------------------------------------")

if __name__ == "__main__":
    try:
        data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        processed_data = preprocess_data(data)
        train_model(processed_data)
    except FileNotFoundError:
        print("Error: The 'WA_Fn-UseC_-Telco-Customer-Churn.csv' file was not found.")
        print("Please download it from Kaggle and place it in the project directory.")
