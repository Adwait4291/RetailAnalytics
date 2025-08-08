# predict.py
import os
import pandas as pd
import numpy as np
import logging
import joblib
import json
import hashlib
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.preprocessing import StandardScaler

# --- Setup and Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_prediction')
load_dotenv()

class Config:
    """Centralized configuration for the prediction pipeline."""
    MLFLOW_TRACKING_URI = "file:///mnt/c/Users/hp/Desktop/Retail-App-Analytics/src/models/mlruns"
    MLFLOW_REGISTERED_MODEL_NAME = "Retail24RandomForestClassifier"
    PRODUCTION_ALIAS = "production"
    BATCH_SIZE = 1000

# --- Context Manager for MongoDB Connection ---
class MongoConnection:
    """A context manager to handle MongoDB connection and disconnection."""
    def __enter__(self):
        logger.info("Connecting to MongoDB...")
        self.client = MongoClient(f"mongodb+srv://{os.getenv('MONGODB_USERNAME')}:{os.getenv('MONGODB_PASSWORD')}@{os.getenv('MONGODB_CLUSTER')}/")
        self.db = self.client[os.getenv("MONGODB_DATABASE")]
        return self.db
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        logger.info("MongoDB connection closed.")

# --- Core Functions ---
def load_production_model_from_mlflow(model_name, alias):
    """Load the production model and associated artifacts directly from MLflow."""
    logger.info(f"Loading production model '{model_name}' with alias '{alias}' from MLflow...")
    try:
        client = MlflowClient()
        prod_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        logger.info(f"Found version {prod_version.version} from run_id {prod_version.run_id}")

        model = mlflow.pyfunc.load_model(prod_version.source)
        local_path = client.download_artifacts(run_id=prod_version.run_id, path=".")

        # Enhanced preprocessor loading with error handling
        scaler_path = os.path.join(local_path, "preprocessor", "preprocessor.pkl")
        
        # Add debugging information
        logger.info(f"Looking for preprocessor at: {scaler_path}")
        logger.info(f"Preprocessor directory exists: {os.path.exists(os.path.dirname(scaler_path))}")
        logger.info(f"Preprocessor file exists: {os.path.exists(scaler_path)}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Preprocessor file not found at {scaler_path}")
            
        scaler = joblib.load(scaler_path)
        
        # CRITICAL: Validate the loaded preprocessor
        logger.info(f"Loaded preprocessor type: {type(scaler)}")
        
        if isinstance(scaler, np.ndarray):
            logger.error("❌ Preprocessor loaded as NumPy array instead of StandardScaler!")
            logger.error("This indicates the preprocessor was saved incorrectly during training.")
            
            # Attempt to reconstruct StandardScaler from the array
            if len(scaler) % 2 == 0:  # Assuming it contains mean and scale parameters
                logger.info("Attempting to reconstruct StandardScaler from NumPy array...")
                n_features = len(scaler) // 2
                reconstructed_scaler = StandardScaler()
                reconstructed_scaler.mean_ = scaler[:n_features]
                reconstructed_scaler.scale_ = scaler[n_features:]
                reconstructed_scaler.n_features_in_ = n_features
                scaler = reconstructed_scaler
                logger.info("✅ Successfully reconstructed StandardScaler from array")
            else:
                raise ValueError("Cannot reconstruct StandardScaler from the loaded array")
        
        elif not hasattr(scaler, 'transform'):
            raise ValueError(f"Loaded object {type(scaler)} does not have a 'transform' method")
        
        # Load scaling info
        scaling_info_path = os.path.join(local_path, "scaling_info.json")
        if not os.path.exists(scaling_info_path):
            raise FileNotFoundError(f"Scaling info not found at {scaling_info_path}")
            
        with open(scaling_info_path, 'r') as f:
            scaling_info = json.load(f)
        
        feature_names = scaling_info.get("all_features")
        if not feature_names: 
            raise ValueError("'all_features' not found in scaling_info.json")

        logger.info("✅ Successfully loaded model and all artifacts from MLflow.")
        return model, scaler, feature_names, scaling_info, prod_version.version
        
    except (MlflowException, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load production model from MLflow: {e}", exc_info=True)
        raise

def fetch_prediction_data(batch_size=Config.BATCH_SIZE):
    """Fetch data from the prediction_data collection."""
    logger.info("Fetching data for prediction...")
    with MongoConnection() as db:
        query = {"predicted": {"$ne": True}}
        total_records = db.prediction_data.count_documents(query)
        logger.info(f"Found {total_records} records requiring prediction.")
        if total_records == 0: return None, 0
        
        cursor = db.prediction_data.find(query)
        df = pd.DataFrame(list(cursor))
        return df, total_records

def process_features(df, feature_names, scaling_info, scaler):
    """Process raw data into features ready for the model."""
    logger.info(f"Processing {len(df)} records into features...")
    df_processed = df.copy()

    # Create composite scores
    if all(c in df_processed for c in ['session_count', 'total_screens_viewed']):
        df_processed['engagement_score'] = (
            df_processed['session_count'] * 0.3 + 
            df_processed.get('used_search_feature', 0) * 0.2 +
            df_processed.get('wrote_review', 0) * 0.15 +
            df_processed.get('added_to_wishlist', 0) * 0.15 +
            df_processed['total_screens_viewed'] * 0.2
        )
    
    # One-hot encode categorical features
    for col in ['region', 'acquisition_channel']:
        if col in df_processed.columns:
            dummies = pd.get_dummies(df_processed[col], prefix=col, dummy_na=False)
            df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)

    # Align columns with model's expected features
    df_processed.columns = df_processed.columns.str.lower()
    missing_features = set(feature_names) - set(df_processed.columns)
    for feat in missing_features:
        df_processed[feat] = 0
    
    # Ensure correct order and scale
    X = df_processed[feature_names].copy()
    scaled_features = [col for col in scaling_info.get("scaled_features", []) if col in X.columns]
    
    if scaled_features:
        logger.info(f"Applying scaling to {len(scaled_features)} features using {type(scaler)}")
        
        # Additional validation before transformation
        if not hasattr(scaler, "transform"):
            raise AttributeError(f"🚨 Scaler object {type(scaler)} does not have 'transform' method")
        
        try:
            # Apply scaling
            X.loc[:, scaled_features] = scaler.transform(X[scaled_features])
            logger.info(f"✅ Successfully scaled {len(scaled_features)} features.")
        except Exception as e:
            logger.error(f"Error during feature scaling: {e}")
            raise
        
    return X

def make_predictions(model, X):
    """Generate predictions and probabilities."""
    logger.info(f"Generating predictions for {len(X)} records...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities

def save_predictions(df, preds, probs, model_version, prediction_id):
    """Save prediction results and metadata back to MongoDB."""
    logger.info(f"Saving {len(df)} predictions to MongoDB...")
    df['purchase_24h_prediction'] = preds
    df['purchase_24h_probability'] = probs
    df['prediction_id'] = prediction_id
    df['model_version'] = model_version
    df['prediction_timestamp'] = datetime.now()

    with MongoConnection() as db:
        # Insert results and update original records
        original_ids = df['_id'].tolist()
        df.drop('_id', axis=1, inplace=True)
        db.predicted_results.insert_many(df.to_dict('records'))
        db.prediction_data.update_many(
            {"_id": {"$in": original_ids}},
            {"$set": {"predicted": True, "prediction_id": prediction_id}}
        )
        
        # Save metadata for the run
        metadata = {
            'prediction_id': prediction_id, 'model_version': model_version,
            'prediction_time': datetime.now(), 'record_count': len(df),
            'positive_rate': np.mean(preds)
        }
        db.prediction_metadata.insert_one(metadata)
    logger.info("Successfully saved predictions and metadata.")

def main():
    """Main function to orchestrate the prediction pipeline."""
    logger.info("Starting prediction pipeline...")
    try:
        # Step 1: Load production model and artifacts from MLflow
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        model, scaler, f_names, s_info, m_version = load_production_model_from_mlflow(
            Config.MLFLOW_REGISTERED_MODEL_NAME, Config.PRODUCTION_ALIAS
        )

        # Step 2: Fetch data needing prediction
        df_raw, total_records = fetch_prediction_data()
        if df_raw is None:
            logger.info("Pipeline finished: No data to predict.")
            return True

        # Step 3: Process features and make predictions
        X_processed = process_features(df_raw, f_names, s_info, scaler)
        predictions, probabilities = make_predictions(model, X_processed)

        # Step 4: Save everything back to the database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_id = f"pred_{timestamp}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
        save_predictions(df_raw, predictions, probabilities, m_version, pred_id)

        logger.info(f"Prediction pipeline completed successfully for {total_records} records.")
        return True
    except Exception as e:
        logger.critical(f"Prediction pipeline failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    exit_code = 0 if main() else 1
    logger.info(f"Exiting with code: {exit_code}")
    exit(exit_code)