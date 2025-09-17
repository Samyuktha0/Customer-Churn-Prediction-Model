# Customer Churn Prediction Model

## Project Overview

This project is a machine learning solution designed to predict customer churn. By analyzing customer data, the model identifies key behavioral patterns and characteristics that indicate a high probability of a customer leaving the service. The goal is to provide a business with an early warning system to help them proactively retain at-risk customers.

This project showcases fundamental skills in data analysis, machine learning modeling, and interpreting a model's performance to provide actionable business insights.

## Key Features

-   **Data Analysis and Preprocessing:** Handles data cleaning, feature engineering, and one-hot encoding for categorical variables.
-   **Machine Learning Modeling:** Trains a classification model (Random Forest Classifier) to predict churn.
-   **Model Evaluation:** Uses a classification report to evaluate the model's accuracy, precision, recall, and F1-score.

## Technologies Used

-   **Python**
-   **Libraries:** pandas, scikit-learn, matplotlib, seaborn

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/churn-prediction.git](https://github.com/your-username/churn-prediction.git)
    cd churn-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the script:**
    ```bash
    python churn_prediction.py
    ```

The script will load the data, train the model, and print the classification report directly to your terminal.

## Data Source

This project uses the Telco Customer Churn dataset, which is publicly available on Kaggle. You will need to download `telco-customer-churn-dataset.zip`, unzip it, and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in your project's root directory.
