# single_prediction.py
# Module for handling single record predictions

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

# --- Add MLflow imports ---
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('retail_single_prediction')
load_dotenv()

# --- Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mnt/c/Users/hp/Desktop/Retail-App-Analytics/src/models/mlruns")
MODEL_NAME = "Retail24RandomForestClassifier"
PRODUCTION_ALIAS = "production"

def connect_to_mongodb():
    """Connect to MongoDB and return database connection."""
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    cluster = os.getenv("MONGODB_CLUSTER")
    database = os.getenv("MONGODB_DATABASE")
    
    if not all([username, password, cluster, database]):
        logger.warning("MongoDB credentials not complete. Some features may not work.")
        return None, None
    
    connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
    try:
        client = MongoClient(connection_string)
        db = client.get_database(database)
        return db, client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        return None, None

def get_active_model_artifacts_from_mlflow():
    """
    Load the production model and its artifacts directly from MLflow Model Registry.
    This is the single source of truth for model loading.
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        logger.info(f"Loading production model '{MODEL_NAME}' from MLflow...")
        
        # Get production model version
        try:
            prod_version = client.get_model_version_by_alias(name=MODEL_NAME, alias=PRODUCTION_ALIAS)
            logger.info(f"Found production version: {prod_version.version}")
        except MlflowException as e:
            logger.error(f"No production model found with alias '{PRODUCTION_ALIAS}': {e}")
            return None, None, None, None
        
        # Load the model
        model_uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ Model loaded successfully")
        
        # Download artifacts
        run_id = prod_version.run_id
        logger.info(f"Downloading artifacts from run: {run_id}")
        
        # Download preprocessor
        try:
            local_path = client.download_artifacts(run_id=run_id, path="preprocessor")
            scaler_path = os.path.join(local_path, "preprocessor.pkl")
            
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Preprocessor not found at {scaler_path}")
            
            scaler = joblib.load(scaler_path)
            logger.info(f"✅ Preprocessor loaded: {type(scaler)}")
            
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            return None, None, None, None
        
        # Download scaling info
        try:
            scaling_info_path = client.download_artifacts(run_id=run_id, path="scaling_info.json")
            with open(scaling_info_path, 'r') as f:
                scaling_info = json.load(f)
            
            feature_names = scaling_info.get("all_features")
            if not feature_names:
                raise ValueError("'all_features' not found in scaling_info.json")
            
            logger.info(f"✅ Scaling info loaded: {len(feature_names)} features")
            
        except Exception as e:
            logger.error(f"Failed to load scaling info: {e}")
            return None, None, None, None
        
        return model, scaler, feature_names, scaling_info
        
    except Exception as e:
        logger.error(f"Error loading model artifacts from MLflow: {str(e)}", exc_info=True)
        return None, None, None, None

def create_feature_df(feature_dict):
    """Convert feature dictionary to a DataFrame."""
    return pd.DataFrame([feature_dict])

def process_single_record(df, feature_names, scaling_info, scaler):
    """
    Process a single record (or batch) for prediction using the same logic as training.
    This function handles both single records and batch processing.
    """
    logger.info(f"Processing {len(df)} record(s) for prediction...")
    df_processed = df.copy()
    
    # === DEBUGGING: Check for initial NaNs ===
    initial_nans = df_processed.isnull().sum()
    if initial_nans.sum() > 0:
        logger.warning(f"Initial NaN values found: {initial_nans[initial_nans > 0].to_dict()}")
    
    # Time-based Processing
    if 'first_visit_date' in df_processed.columns:
        df_processed['first_visit_date'] = pd.to_datetime(df_processed['first_visit_date'], errors='coerce')
        # FIX: Handle NaT (Not a Time) values from failed datetime parsing
        df_processed['hour'] = df_processed['first_visit_date'].dt.hour.fillna(12) # Default to noon
        df_processed['dayofweek'] = df_processed['first_visit_date'].dt.dayofweek.fillna(0) # Default to Monday
        df_processed['is_weekend'] = df_processed['dayofweek'].isin([5,6]).astype(int)
    
    # Screen List Processing
    if 'screen_list' in df_processed.columns:
        # FIX: Handle missing screen_list values
        df_processed['screen_list'] = df_processed['screen_list'].fillna('').astype(str) + ','
        
        # Define screen categories (same as in processing.py)
        shopping_screens = ['ProductList', 'ProductDetail', 'CategoryBrowse', 'Search']
        cart_screens = ['ShoppingCart', 'Checkout', 'PaymentMethods', 'DeliveryOptions']
        engagement_screens = ['WishList', 'Reviews', 'Promotions']
        account_screens = ['Account', 'AddressBook', 'OrderTracking']
        
        # Create binary indicators for each screen
        all_tracked_screens = shopping_screens + cart_screens + engagement_screens + account_screens
        for screen in all_tracked_screens:
            df_processed[screen.lower()] = df_processed['screen_list'].str.contains(screen, na=False).astype(int)
        
        # Create count features for each category
        df_processed['shopping_count'] = df_processed[[s.lower() for s in shopping_screens]].sum(axis=1)
        df_processed['cart_count'] = df_processed[[s.lower() for s in cart_screens]].sum(axis=1)
        df_processed['engagement_count'] = df_processed[[s.lower() for s in engagement_screens]].sum(axis=1)
        df_processed['account_count'] = df_processed[[s.lower() for s in account_screens]].sum(axis=1)
        
        # Create other screens count - FIX: Handle empty strings properly
        df_processed['other_screens'] = df_processed['screen_list'].apply(
            lambda x: len([s for s in str(x).split(',') if s and s.strip() and s not in all_tracked_screens]) if pd.notna(x) else 0
        )
    
    # Feature Engineering - FIX: Add fillna for all calculations
    if all(col in df_processed.columns for col in ['session_count', 'used_search_feature', 'wrote_review', 'added_to_wishlist', 'total_screens_viewed']):
        # Fill NaN values before calculation
        df_processed['session_count'] = df_processed['session_count'].fillna(0)
        df_processed['used_search_feature'] = df_processed['used_search_feature'].fillna(0)
        df_processed['wrote_review'] = df_processed['wrote_review'].fillna(0)
        df_processed['added_to_wishlist'] = df_processed['added_to_wishlist'].fillna(0)
        df_processed['total_screens_viewed'] = df_processed['total_screens_viewed'].fillna(0)
        
        df_processed['engagement_score'] = (
            df_processed['session_count'] * 0.3 +
            df_processed['used_search_feature'] * 0.2 +
            df_processed['wrote_review'] * 0.15 +
            df_processed['added_to_wishlist'] * 0.15 +
            df_processed['total_screens_viewed'] * 0.2
        )
    
    if all(col in df_processed.columns for col in ['shopping_count', 'cart_count', 'engagement_count', 'account_count']):
        df_processed['screen_diversity'] = (
            df_processed[['shopping_count', 'cart_count', 'engagement_count', 'account_count']].gt(0).sum(axis=1)
        )
        
        if 'added_to_wishlist' in df_processed.columns:
            df_processed['purchase_intent'] = (
                df_processed['cart_count'] * 0.4 +
                df_processed['shopping_count'] * 0.3 +
                df_processed['engagement_count'] * 0.2 +
                df_processed['added_to_wishlist'] * 0.1
            )
    
    # Categorical Feature Processing - FIX: Handle unknown categories
    if 'platform' in df_processed.columns:
        df_processed['platform'] = df_processed['platform'].map({'iOS': 1, 'Android': 0}).fillna(0)
    
    # Region encoding - FIX: Handle missing regions
    if 'region' in df_processed.columns:
        df_processed['region'] = df_processed['region'].fillna('Unknown')
        region_dummies = pd.get_dummies(df_processed['region'], prefix='region')
        df_processed = pd.concat([df_processed, region_dummies], axis=1)
    
    # Acquisition channel encoding - FIX: Handle missing channels
    if 'acquisition_channel' in df_processed.columns:
        df_processed['acquisition_channel'] = df_processed['acquisition_channel'].fillna('Unknown')
        channel_dummies = pd.get_dummies(df_processed['acquisition_channel'], prefix='channel')
        df_processed = pd.concat([df_processed, channel_dummies], axis=1)
    
    # User segment processing - FIX: Handle missing segments
    if 'user_segment' in df_processed.columns:
        df_processed['user_segment'] = df_processed['user_segment'].fillna('Unknown User')
        df_processed['age_group'] = df_processed['user_segment'].apply(lambda x: str(x).split()[0] if pd.notna(x) else 'Unknown')
        df_processed['user_type'] = df_processed['user_segment'].apply(lambda x: ' '.join(str(x).split()[1:]) if pd.notna(x) and len(str(x).split()) > 1 else 'User')
        
        age_group_dummies = pd.get_dummies(df_processed['age_group'], prefix='age_group')
        user_type_dummies = pd.get_dummies(df_processed['user_type'], prefix='user_type')
        df_processed = pd.concat([df_processed, age_group_dummies, user_type_dummies], axis=1)
    
    # App version processing - FIX: Handle invalid versions
    if 'app_version' in df_processed.columns:
        def safe_version_parse(version):
            try:
                if pd.isna(version):
                    return 3 # Default major version
                return int(str(version).split('.')[0])
            except:
                return 3
        
        def safe_version_score(version):
            try:
                if pd.isna(version):
                    return 3.0 # Default score
                return sum(float(i)/(10**n) for n, i in enumerate(str(version).split('.')))
            except:
                return 3.0
        
        df_processed['app_major_version'] = df_processed['app_version'].apply(safe_version_parse)
        df_processed['version_score'] = df_processed['app_version'].apply(safe_version_score)
    
    # FIX: Fill any remaining numeric NaNs
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
    
    # Clean up - drop original columns that have been processed
    columns_to_drop = [
        col for col in ['screen_list', 'first_visit_date', 'region', 'acquisition_channel', 
                       'user_segment', 'app_version', 'age_group', 'user_type'] 
        if col in df_processed.columns
    ]
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Ensure all column names are lowercase
    df_processed.columns = df_processed.columns.str.lower()
    
    # Add missing features as zeros
    missing_features = [f for f in feature_names if f not in df_processed.columns]
    for feat in missing_features:
        df_processed[feat] = 0
    
    # Select only the features the model expects, in the correct order
    X = df_processed[feature_names].copy()
    
    # === FINAL NaN CHECK AND FIX ===
    final_nans = X.isnull().sum()
    if final_nans.sum() > 0:
        logger.warning(f"NaN values found before scaling: {final_nans[final_nans > 0].to_dict()}")
        # Fill any remaining NaNs with 0
        X = X.fillna(0)
        logger.info("✅ Filled remaining NaN values with 0")
    
    # Apply scaling to the features that were scaled during training
    scaled_features = scaling_info.get("scaled_features", [])
    if scaled_features and scaler is not None:
        scale_cols = [col for col in scaled_features if col in X.columns]
        if scale_cols:
            logger.info(f"Applying scaling to {len(scale_cols)} features")
            # FIX: Ensure no NaNs in scaled columns
            X[scale_cols] = X[scale_cols].fillna(0)
            X[scale_cols] = scaler.transform(X[scale_cols])
    
    # === FINAL VERIFICATION ===
    if X.isnull().sum().sum() > 0:
        logger.error("❌ NaN values still present after processing!")
        X = X.fillna(0) # Final safety net
        logger.info("✅ Applied final NaN fill")
    
    logger.info(f"✅ Processing complete. Final shape: {X.shape}")
    return X

def make_single_prediction(feature_dict):
    """Make a prediction for a single record."""
    logger.info("Making single prediction...")
    
    model, scaler, feature_names, scaling_info = get_active_model_artifacts_from_mlflow()
    
    if model is None:
        return {"error": "Failed to load model artifacts from MLflow.", "success": False}
    
    try:
        df = create_feature_df(feature_dict)
        X = process_single_record(df, feature_names, scaling_info, scaler)
        
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])
        
        feature_influences = {}
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
            
            # Add feature values
            for feature in feature_names:
                # Get the value, defaulting to 0 if the column doesn't exist
                value = X[feature].values[0] if feature in X.columns else 0
                # FIX: Explicitly cast the value to float to avoid dtype warnings
                features_df.loc[features_df['feature'] == feature, 'value'] = float(value)

            # Calculate influence (importance * value)
            features_df['influence'] = features_df['importance'] * features_df['value']
            features_df['abs_influence'] = features_df['influence'].abs()
            features_df = features_df.sort_values('abs_influence', ascending=False)
            
            feature_influences = {
                'top_features': features_df.head(5).to_dict('records'),
                'all_features': features_df.to_dict('records')
            }
        
        result = {
            "prediction": prediction,
            "probability": probability,
            "feature_influences": feature_influences,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        logger.info(f"✅ Prediction complete: {prediction} (prob: {probability:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        return {"error": str(e), "success": False}

def save_prediction_to_mongodb(feature_dict, prediction_result):
    """Save the prediction to MongoDB (optional)."""
    try:
        db, client = connect_to_mongodb()
        if db is None: 
            logger.warning("MongoDB not available. Skipping save.")
            return False
        
        record = {**feature_dict}
        record['purchase_24h_prediction'] = prediction_result['prediction']
        record['purchase_24h_probability'] = prediction_result['probability']
        record['prediction_timestamp'] = datetime.now()
        record['prediction_source'] = 'single_prediction_ui'
        
        db.predicted_results.insert_one(record)
        client.close()
        logger.info("✅ Prediction saved to MongoDB")
        return True
        
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        if 'client' in locals() and client: 
            client.close()
        return False

# Example usage (for testing)
if __name__ == "__main__":
    sample_features = {
        "user_id": "test_user_1", 
        "platform": "iOS", 
        "age": 35, 
        "session_count": 5,
        "total_screens_viewed": 15, 
        "used_search_feature": 1, 
        "wrote_review": 0, 
        "added_to_wishlist": 1,
        "screen_list": "ProductList,ProductDetail,ShoppingCart,Checkout", 
        "region": "NorthAmerica",
        "acquisition_channel": "Organic", 
        "user_segment": "Young Professional", 
        "app_version": "3.2.1",
        "first_visit_date": "2025-05-01"
    }
    
    result = make_single_prediction(sample_features)
    print(json.dumps(result, indent=2, default=str))

# Additional debugging function you can add:
def debug_nan_sources(df):
    """Debug function to identify NaN sources in your data"""
    print("=== NaN Analysis ===")
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        print("Columns with NaN values:")
        for col, count in nan_cols.items():
            print(f"  {col}: {count} NaN values ({count/len(df)*100:.1f}%)")
            
        print("\nSample rows with NaN values:")
        nan_mask = df.isnull().any(axis=1)
        print(df[nan_mask].head())
    else:
        print("No NaN values found!")
    
    return nan_cols