# train.py - Complete Robust ML Training Pipeline
"""
A comprehensive machine learning training pipeline that combines:
- Intelligent retraining logic based on data freshness and model age
- Robust data preprocessing with missing value handling
- MLflow integration for experiment tracking and model management
- MongoDB integration for data retrieval and metadata storage
- Automated model promotion based on performance improvements
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
import json
import tempfile
import hashlib
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

# Database and ML imports
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.datasets import make_classification

# MLflow imports for modern MLOps workflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- 1. Centralized Configuration ---
class Config:
    """Centralized configuration for the ML pipeline."""
    
    # Environment Detection
    IS_DOCKER = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mnt/c/Users/hp/Desktop/Retail-App-Analytics/src/models/mlruns")
    MLFLOW_EXPERIMENT_NAME = "Retail24_Training_Experiment"
    MLFLOW_REGISTERED_MODEL_NAME = "Retail24RandomForestClassifier"
    PRODUCTION_ALIAS = "production"
    
    # Model Directory (environment-aware)
    MODEL_DIR = "/app/models" if IS_DOCKER else os.getenv("MODEL_DIR", "../models")
    
    # Model Training Parameters
    N_ESTIMATORS = 100
    MAX_DEPTH = None
    MIN_SAMPLES_SPLIT = 2
    MIN_SAMPLES_LEAF = 1
    MAX_FEATURES = 'sqrt'
    CLASS_WEIGHT = 'balanced'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Retraining Thresholds
    MIN_NEW_RECORDS_THRESHOLD = 200
    MIN_PERFORMANCE_IMPROVEMENT = 0.01  # 1% improvement required for promotion
    MAX_MODEL_AGE_DAYS = 30
    
    # Data Quality Parameters
    MAX_MISSING_PERCENTAGE = 0.3  # Drop columns with >30% missing values
    SYNTHETIC_DATA_SAMPLES = 1000  # For development fallback

# --- 2. Setup and Logging ---
def setup_logging():
    """Configure logging with appropriate format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('retail_ml_training')

logger = setup_logging()
load_dotenv()

# --- 3. Database Connection and Data Handling ---
def connect_to_mongodb() -> object:
    """Connect to MongoDB and return the database object."""
    try:
        username = os.getenv("MONGODB_USERNAME")
        password = os.getenv("MONGODB_PASSWORD")
        cluster = os.getenv("MONGODB_CLUSTER")
        database = os.getenv("MONGODB_DATABASE")
        
        if not all([username, password, cluster, database]):
            raise ValueError("Missing MongoDB credentials in environment variables")
        
        connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
        client = MongoClient(connection_string)
        db = client.get_database(database)
        
        # Test connection
        db.list_collection_names()
        logger.info("✅ Successfully connected to MongoDB")
        return db
        
    except Exception as e:
        logger.error(f"❌ Error connecting to MongoDB: {e}")
        raise

def get_all_processed_data(db) -> Tuple[pd.DataFrame, str]:
    """
    Get all processed data from MongoDB.
    If no data is found, generate synthetic data for development.
    """
    logger.info("Retrieving all processed data...")
    
    try:
        cursor = db.processed_retail_data.find({}, {'_id': 0})
        df = pd.DataFrame(list(cursor))
        
        if df.empty:
            logger.warning("⚠️ No processed data found. Generating synthetic data for development.")
            X, y = make_classification(
                n_samples=Config.SYNTHETIC_DATA_SAMPLES,
                n_features=20,
                n_informative=10,
                n_redundant=5,
                n_classes=2,
                random_state=Config.RANDOM_STATE,
                flip_y=0.05
            )
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            df['purchase_24h'] = y
            latest_version = "synthetic-0.1.0"
        else:
            # Get latest processing version
            metadata = db.processing_metadata.find_one(
                {"domain": "retail"}, 
                sort=[("processed_at", -1)]
            )
            latest_version = metadata.get("processing_version", "unknown") if metadata else "unknown"
        
        logger.info(f"Retrieved {len(df)} records. Latest data version: {latest_version}")
        return df, latest_version
        
    except Exception as e:
        logger.error(f"Error retrieving processed data: {e}")
        raise

def prepare_data_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, Dict[str, Any]]:
    """
    Comprehensive data preparation including cleaning, imputation, splitting, and scaling.
    """
    logger.info("Starting comprehensive data preparation...")
    original_shape = df.shape
    
    # --- Step 1: Remove identifier and non-feature columns ---
    cols_to_drop = ['user_id', 'record_hash', 'processing_version', '_id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # --- Step 2: Handle timestamp columns ---
    timestamp_columns = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.contains(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', na=False).any():
            timestamp_columns.append(col)
    
    if timestamp_columns:
        logger.info(f"Dropping detected timestamp columns: {timestamp_columns}")
        df = df.drop(columns=timestamp_columns)
    
    # --- Step 3: Handle remaining object columns ---
    object_columns = df.select_dtypes(include=['object']).columns
    if not object_columns.empty:
        logger.warning(f"Dropping non-numeric columns: {object_columns.tolist()}")
        df = df.drop(columns=object_columns)
    
    # --- Step 4: Comprehensive missing value handling ---
    logger.info("Analyzing missing values...")
    missing_stats = df.isnull().sum()
    missing_percentages = (missing_stats / len(df)) * 100
    
    # Drop columns with excessive missing values
    cols_to_drop_missing = missing_percentages[missing_percentages > Config.MAX_MISSING_PERCENTAGE * 100].index
    if len(cols_to_drop_missing) > 0:
        logger.warning(f"Dropping columns with >{Config.MAX_MISSING_PERCENTAGE*100}% missing values: {cols_to_drop_missing.tolist()}")
        df = df.drop(columns=cols_to_drop_missing)
    
    # Impute remaining missing values
    if df.isnull().sum().sum() > 0:
        logger.info("Imputing remaining missing values...")
        
        # Numeric columns: use median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {missing_stats[col]} missing values in '{col}' with median: {median_val:.3f}")
        
        # Boolean columns: use False
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(False)
                logger.info(f"Filled {missing_stats[col]} missing values in '{col}' with False")
    
    # --- Step 5: Validate target column ---
    if 'purchase_24h' not in df.columns:
        raise ValueError("Target column 'purchase_24h' not found in dataset")
    
    # --- Step 6: Prepare features and target ---
    X = df.drop('purchase_24h', axis=1)
    y = df['purchase_24h']
    
    logger.info(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
    
    # Validate we have enough data
    if len(X) < 100:
        raise ValueError(f"Insufficient data for training: {len(X)} samples")
    
    # --- Step 7: Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, 
        stratify=y
    )
    
    # --- Step 8: Feature scaling ---
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Scaling {len(numeric_features)} numeric features")
    
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    if numeric_features:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
        
        X_train = X_train_scaled
        X_test = X_test_scaled
    
    # Store metadata
    scaling_info = {
        "scaled_features": numeric_features,
        "all_features": X.columns.tolist(),
        "original_shape": list(original_shape),
        "final_shape": list(X.shape),
        "target_distribution": y.value_counts(normalize=True).to_dict()
    }
    
    logger.info(f"Data preparation complete:")
    logger.info(f"  Original shape: {original_shape}")
    logger.info(f"  Final shape: {X.shape}")
    logger.info(f"  Training set: {X_train.shape[0]} samples")
    logger.info(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler, scaling_info

# --- 4. Model Training and Evaluation ---
def train_and_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[RandomForestClassifier, Dict[str, float], np.ndarray, pd.DataFrame]:
    """Train the RandomForest model and return comprehensive evaluation results."""
    logger.info("Training and evaluating Random Forest model...")
    
    # Initialize model with robust parameters
    model = RandomForestClassifier(
        n_estimators=Config.N_ESTIMATORS,
        max_depth=Config.MAX_DEPTH,
        min_samples_split=Config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=Config.MIN_SAMPLES_LEAF,
        max_features=Config.MAX_FEATURES,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
        class_weight=Config.CLASS_WEIGHT
    )
    
    # Cross-validation before final training
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=Config.CV_FOLDS, 
        scoring='f1_weighted',
        n_jobs=-1
    )
    logger.info(f"CV F1 scores: {cv_scores}")
    logger.info(f"Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    metrics = {
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "test_f1_score": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_pred_proba_test))
    }
    
    logger.info("Model Performance Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate additional evaluation artifacts
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Feature importance analysis
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Classification report
    class_report = classification_report(y_test, y_pred_test, output_dict=True)
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred_test)}")
    
    return model, metrics, conf_matrix, importance_df

# --- 5. MLflow and Model Management ---
def get_production_model_metrics(client: MlflowClient, model_name: str) -> Tuple[Optional[str], float]:
    """Get the production model's F1 score and run_id using the 'production' alias."""
    try:
        prod_version = client.get_model_version_by_alias(model_name, Config.PRODUCTION_ALIAS)
        run = client.get_run(prod_version.run_id)
        f1_score_val = run.data.metrics.get("test_f1_score", 0.0)
        logger.info(f"Found production model: Version {prod_version.version} (Run ID: {prod_version.run_id}) with Test F1: {f1_score_val:.4f}")
        return prod_version.run_id, f1_score_val
    except MlflowException as e:
        logger.info(f"No production model found with '{Config.PRODUCTION_ALIAS}' alias: {e}")
        return None, 0.0

def should_train_model(db, prod_run_id: Optional[str]) -> Tuple[bool, str]:
    """
    Determine if a new model should be trained based on:
    1. Existence of production model
    2. Model age
    3. Amount of new data available
    """
    if not prod_run_id:
        return True, "Initial model training (no production model found)."
    
    try:
        # Get production model training time
        run = MlflowClient().get_run(prod_run_id)
        last_train_time = datetime.fromtimestamp(run.info.start_time / 1000)
        
        # Check model age
        days_since_training = (datetime.now() - last_train_time).days
        if days_since_training > Config.MAX_MODEL_AGE_DAYS:
            return True, f"Model is {days_since_training} days old (threshold: {Config.MAX_MODEL_AGE_DAYS} days)."
        
        # Check for new data since last training
        logger.info(f"Checking for new data since: {last_train_time}")
        new_data_cursor = db.processing_metadata.find({
            "processed_at": {"$gt": last_train_time}, 
            "domain": "retail"
        })
        
        new_records_count = sum(entry.get('record_count', 0) for entry in new_data_cursor)
        logger.info(f"Found {new_records_count} new records since last training")
        
        if new_records_count >= Config.MIN_NEW_RECORDS_THRESHOLD:
            return True, f"Sufficient new data available ({new_records_count} records, threshold: {Config.MIN_NEW_RECORDS_THRESHOLD})."
        
        return False, f"Training not required. Model age: {days_since_training} days, New data: {new_records_count} records (below threshold)."
        
    except Exception as e:
        logger.warning(f"Error checking training requirements: {e}. Proceeding with training.")
        return True, "Error in training decision logic, proceeding with training for safety."

def log_artifacts_safely(scaler: StandardScaler, confusion_matrix_data: np.ndarray, feature_importance_df: pd.DataFrame, scaling_info_dict: Dict[str, Any]):
    """Save artifacts to temporary directory and log them to MLflow."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info("Logging artifacts to MLflow...")
            
            # Save preprocessor in dedicated directory
            preprocessor_dir = os.path.join(temp_dir, "preprocessor")
            os.makedirs(preprocessor_dir)
            joblib.dump(scaler, os.path.join(preprocessor_dir, "preprocessor.pkl"))
            mlflow.log_artifacts(preprocessor_dir, artifact_path="preprocessor")
            
            # Save confusion matrix
            conf_matrix_path = os.path.join(temp_dir, "confusion_matrix.json")
            with open(conf_matrix_path, "w") as f:
                json.dump(confusion_matrix_data.tolist(), f, indent=2)
            mlflow.log_artifact(conf_matrix_path)
            
            # Save feature importance
            feat_importance_path = os.path.join(temp_dir, "feature_importance.json")
            feature_importance_df.to_json(feat_importance_path, orient="records", indent=2)
            mlflow.log_artifact(feat_importance_path)
            
            # Save scaling information
            scaling_info_path = os.path.join(temp_dir, "scaling_info.json")
            with open(scaling_info_path, "w") as f:
                json.dump(scaling_info_dict, f, indent=2)
            mlflow.log_artifact(scaling_info_path)
            
            logger.info("✅ All artifacts successfully logged to MLflow")
            
    except Exception as e:
        logger.error(f"Error logging artifacts: {e}")
        raise

def save_legacy_model_files(model: RandomForestClassifier, scaler: StandardScaler, model_version: str, metrics: Dict[str, float], scaling_info: Dict[str, Any], data_version: str, db):
    """Save model files locally and metadata to MongoDB for backward compatibility."""
    try:
        logger.info(f"Saving legacy model files for version: {model_version}")
        
        # Create model directory
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        
        # Define file paths
        model_path = os.path.join(Config.MODEL_DIR, f"rf_model_{model_version}.pkl")
        scaler_path = os.path.join(Config.MODEL_DIR, f"scaler_{model_version}.pkl")
        feature_names_path = os.path.join(Config.MODEL_DIR, f"feature_names_{model_version}.json")
        scaling_info_path = os.path.join(Config.MODEL_DIR, f"scaling_info_{model_version}.json")
        
        # Save files
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        with open(feature_names_path, 'w') as f:
            json.dump(scaling_info["all_features"], f, indent=2)
        
        with open(scaling_info_path, 'w') as f:
            json.dump(scaling_info, f, indent=2)
        
        logger.info(f"Saved model files to {Config.MODEL_DIR}")
        
        # Save metadata to MongoDB
        try:
            # Deactivate existing models
            db.model_metadata.update_many(
                {"status": "active"},
                {"$set": {"status": "inactive"}}
            )
            
            # Insert new metadata
            metadata = {
                "model_version": model_version,
                "model_type": "RandomForest",
                "trained_at": datetime.now(),
                "data_version": data_version,
                "metrics": metrics,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "feature_names_path": feature_names_path,
                "scaling_info_path": scaling_info_path,
                "status": "active",
                "feature_count": len(scaling_info["all_features"]),
                "scaled_feature_count": len(scaling_info["scaled_features"])
            }
            
            db.model_metadata.insert_one(metadata)
            logger.info("✅ Model metadata saved to MongoDB")
            
        except Exception as e:
            logger.warning(f"Could not save metadata to MongoDB: {e}")
            
    except Exception as e:
        logger.error(f"Error saving legacy model files: {e}")
        # Don't raise - this is for backward compatibility only

# --- 6. Main Orchestration ---
def main():
    """Main function to orchestrate the model training and promotion pipeline."""
    logger.info("🚀 Starting comprehensive model training pipeline")
    
    try:
        # --- Setup MLflow ---
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        try:
            experiment_id = mlflow.create_experiment(
                Config.MLFLOW_EXPERIMENT_NAME, 
                tags={"project": "Retail_App_Analytics", "version": "2.0"}
            )
            logger.info(f"Created MLflow experiment: {Config.MLFLOW_EXPERIMENT_NAME} (ID: {experiment_id})")
        except MlflowException:
            logger.info(f"Using existing MLflow experiment: {Config.MLFLOW_EXPERIMENT_NAME}")
        
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        client = MlflowClient()
        
        # --- Setup database connection ---
        db = connect_to_mongodb()
        
        # --- Training Decision Logic ---
        prod_run_id, best_f1_score = get_production_model_metrics(client, Config.MLFLOW_REGISTERED_MODEL_NAME)
        train_needed, reason = should_train_model(db, prod_run_id)
        
        if not train_needed:
            logger.info(f"✅ Pipeline completed early. {reason}")
            return
        
        logger.info(f"Proceeding with training. Reason: {reason}")
        
        # --- Start MLflow Run ---
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run started: {run_id}")
            
            try:
                # --- Data Preparation ---
                df, latest_data_version = get_all_processed_data(db)
                X_train, X_test, y_train, y_test, scaler, scaling_info = prepare_data_for_training(df)
                
                # --- Model Training and Evaluation ---
                model, new_metrics, conf_matrix, feat_importance = train_and_evaluate_model(
                    X_train, y_train, X_test, y_test
                )
                new_f1_score = new_metrics.get("test_f1_score", 0.0)
                
                # --- MLflow Logging ---
                # Log parameters
                mlflow.log_params({
                    "n_estimators": Config.N_ESTIMATORS,
                    "max_features": Config.MAX_FEATURES,
                    "class_weight": Config.CLASS_WEIGHT,
                    "data_version": latest_data_version,
                    "feature_count": X_train.shape[1],
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "training_reason": reason
                })
                
                # Log metrics
                mlflow.log_metrics(new_metrics)
                
                # Log artifacts
                log_artifacts_safely(scaler, conf_matrix, feat_importance, scaling_info)
                
                # --- Model Registration and Promotion Logic ---
                signature = infer_signature(X_train, model.predict(X_train))
                improvement = new_f1_score - best_f1_score
                
                if improvement > Config.MIN_PERFORMANCE_IMPROVEMENT:
                    logger.info(f"📈 PROMOTION: New model F1 ({new_f1_score:.4f}) improves upon production F1 ({best_f1_score:.4f}) by {improvement:.4f}")
                    
                    # Register and promote model
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        signature=signature,
                        input_example=X_train.head(1),
                        registered_model_name=Config.MLFLOW_REGISTERED_MODEL_NAME
                    )
                    
                    new_version = model_info.model_version
                    client.set_registered_model_alias(
                        name=Config.MLFLOW_REGISTERED_MODEL_NAME, 
                        alias=Config.PRODUCTION_ALIAS, 
                        version=new_version
                    )
                    logger.info(f"✅ Model version {new_version} promoted to PRODUCTION")
                    
                else:
                    logger.info(f"ℹ️ NO PROMOTION: New model F1 ({new_f1_score:.4f}) improvement ({improvement:.4f}) below threshold ({Config.MIN_PERFORMANCE_IMPROVEMENT})")
                    mlflow.sklearn.log_model(
                        sk_model=model, 
                        artifact_path="model_not_promoted", 
                        signature=signature,
                        input_example=X_train.head(1)
                    )
                
                # --- Legacy Support: Save files and MongoDB metadata ---
                model_version = f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(new_metrics).encode()).hexdigest()[:8]}"
                save_legacy_model_files(model, scaler, model_version, new_metrics, scaling_info, latest_data_version, db)
                
                logger.info(f"🎉 Training pipeline completed successfully!")
                logger.info(f"   New model F1 score: {new_f1_score:.4f}")
                logger.info(f"   Production F1 score: {best_f1_score:.4f}")
                logger.info(f"   Improvement: {improvement:.4f}")
                
            except Exception as e:
                logger.error(f"Error during training execution: {e}", exc_info=True)
                mlflow.log_param("training_status", "failed")
                mlflow.log_param("error_message", str(e))
                raise
                
    except Exception as e:
        logger.critical(f"❌ Critical pipeline failure: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()