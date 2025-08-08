roblem Statement and Solution
The core problem is to identify retail app users most likely to purchase within 24 hours to enable targeted marketing and increase conversion rates. My solution is a full-stack machine learning pipeline that automates data processing, trains a predictive model, and deploys it in a production-ready, containerized environment with continuous monitoring.

Technical Deep Dive
1. Data Processing and Feature Engineering
What: An automated pipeline ingests raw CSV user interaction logs, processes them using Pandas and NumPy, and persists the cleaned data into MongoDB for scalable access.

Why: Storing processed data in a NoSQL database like MongoDB provides flexibility and scalability for handling large, semi-structured user log data. Feature engineering is critical for transforming raw logs into meaningful signals that the model can learn from.

How: The processing.py script generates a rich feature set, including time-based features, behavioral flags (e.g., cart_count, shopping_count), and custom composite scores like a purchase_intent score with heavy weighting on cart interactions.

Python

# Example of feature engineering in processing.py
import pandas as pd

def engineer_features(df):
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.weekday >= 5
    df['cart_count'] = df['screen_list'].apply(lambda x: x.count('cart'))
    return df
2. Model Training and Optimization
What: I trained a Random Forest Classifier using Scikit-learn for the purchase prediction task.

Why: Random Forest was selected for its robustness and ability to handle non-linear relationships in behavioral data. The primary objective was to maximize recall (achieving 1.0) for the positive class (purchasers) to ensure every potential buyer is flagged, even at the cost of some precision. This strategy directly addresses the business goal of minimizing missed sales opportunities.

How: GridSearchCV was used to systematically tune hyperparameters (n_estimators, max_depth) to optimize for the recall metric. MLflow was integrated to log all hyperparameters, metrics (F1-score: 0.91, Recall: 1.0, ROC AUC), and the final serialized model for versioning.

Python

# Example of model training and MLflow logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow

with mlflow.start_run():
    rf_model = RandomForestClassifier()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 10]}
    grid_search = GridSearchCV(rf_model, param_grid, scoring='recall', cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("recall", best_model.score(X_test, y_test))
    mlflow.sklearn.log_model(best_model, "random-forest-model")
3. MLOps and Deployment
What: The entire system is orchestrated by an MLOps pipeline that manages model lifecycle, from training to deployment and monitoring.

Why: To transition from a static model to a reproducible, scalable, and maintainable production system. This ensures the model remains relevant and reliable over time.

How:

Containerization: The application is Dockerized using separate Dockerfiles for the training pipeline and the Streamlit dashboard, ensuring environment consistency.

MLflow Model Registry: Models are registered and versioned in the MLflow Model Registry. All inference services dynamically fetch the latest production model, decoupling deployment from model training.

Automated Retraining: The pipeline includes triggers for retraining based on data drift (e.g., +10% new data) or performance degradation (e.g., -5% F1-score), keeping the model's performance stable.

Deployment: A FastAPI microservice provides a RESTful API for real-time predictions. This service, along with a Streamlit dashboard for monitoring model metrics and handling batch/single predictions, is deployed as a containerized solution.

Python

# Example of loading the production model from MLflow
import mlflow.pyfunc

def load_production_model():
    # Dynamically loads the model with the 'Production' alias
    model_uri = "models:/Retail24RandomForestClassifier@production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

# In the FastAPI endpoint
# model = load_production_model()
# @app.post("/predict")
# def predict(data: dict):
#     prediction = model.predict(pd.DataFrame([data]))
#     return {"prediction": prediction.tolist()}
