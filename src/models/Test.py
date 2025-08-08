# test_pipeline.py - Script to test your fixed ML pipeline

import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import tempfile
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pipeline_test')

def test_preprocessor_saving_and_loading():
    """Test the preprocessor saving and loading functionality."""
    logger.info("Testing preprocessor saving and loading...")
    
    # Create a dummy StandardScaler
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit a scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info(f"Original scaler type: {type(scaler)}")
    logger.info(f"Scaler mean shape: {scaler.mean_.shape}")
    logger.info(f"Scaler scale shape: {scaler.scale_.shape}")
    
    # Test saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        preprocessor_path = os.path.join(temp_dir, "test_preprocessor.pkl")
        
        # Save
        joblib.dump(scaler, preprocessor_path)
        logger.info(f"Saved preprocessor to {preprocessor_path}")
        
        # Load
        loaded_scaler = joblib.load(preprocessor_path)
        logger.info(f"Loaded scaler type: {type(loaded_scaler)}")
        
        # Test functionality
        X_test_scaled_original = scaler.transform(X_test)
        X_test_scaled_loaded = loaded_scaler.transform(X_test)
        
        # Check if results are the same
        import numpy as np
        if np.allclose(X_test_scaled_original, X_test_scaled_loaded):
            logger.info("✅ Preprocessor saving/loading test PASSED")
            return True
        else:
            logger.error("❌ Preprocessor saving/loading test FAILED - results don't match")
            return False

def test_mlflow_artifact_structure():
    """Test the MLflow artifact structure for your model."""
    logger.info("Testing MLflow artifact structure...")
    
    # Set MLflow tracking URI
    mlflow_uri = "file:///mnt/c/Users/hp/Desktop/Retail-App-Analytics/src/models/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        client = MlflowClient()
        
        # Try to get the production model
        model_name = "Retail24RandomForestClassifier"
        try:
            prod_version = client.get_model_version_by_alias(model_name, "production")
            logger.info(f"✅ Found production model: Version {prod_version.version}")
            
            # Download artifacts to check structure
            local_path = client.download_artifacts(run_id=prod_version.run_id, path=".")
            
            # Check for required files
            required_files = [
                "scaling_info.json",
                "preprocessor/preprocessor.pkl"
            ]
            
            missing_files = []
            for file_path in required_files:
                full_path = os.path.join(local_path, file_path)
                if os.path.exists(full_path):
                    logger.info(f"✅ Found required file: {file_path}")
                else:
                    logger.error(f"❌ Missing required file: {file_path}")
                    missing_files.append(file_path)
            
            if not missing_files:
                logger.info("✅ All required artifacts found")
                
                # Test loading the preprocessor
                preprocessor_path = os.path.join(local_path, "preprocessor", "preprocessor.pkl")
                try:
                    loaded_preprocessor = joblib.load(preprocessor_path)
                    logger.info(f"✅ Successfully loaded preprocessor: {type(loaded_preprocessor)}")
                    
                    if isinstance(loaded_preprocessor, StandardScaler):
                        logger.info("✅ Preprocessor is correct type (StandardScaler)")
                        if hasattr(loaded_preprocessor, 'transform'):
                            logger.info("✅ Preprocessor has transform method")
                            return True
                        else:
                            logger.error("❌ Preprocessor missing transform method")
                            return False
                    else:
                        logger.error(f"❌ Preprocessor is wrong type: {type(loaded_preprocessor)}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load preprocessor: {e}")
                    return False
            else:
                logger.error(f"❌ Missing required files: {missing_files}")
                return False
                
        except Exception as e:
            logger.error(f"❌ No production model found or error accessing it: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ MLflow connection error: {e}")
        return False

def run_full_pipeline_test():
    """Run the full pipeline to test everything works."""
    logger.info("Running full pipeline test...")
    
    try:
        # Import your training module
        # Adjust the import path based on your project structure
        from train import main as train_main
        
        logger.info("Starting training pipeline...")
        train_main()
        logger.info("✅ Training pipeline completed successfully")
        
        # Test prediction pipeline
        from predict import main as predict_main
        
        logger.info("Starting prediction pipeline...")
        predict_main()
        logger.info("✅ Prediction pipeline completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}", exc_info=True)
        return False

def main():
    """Run all tests."""
    logger.info("Starting ML Pipeline Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Preprocessor Save/Load Test", test_preprocessor_saving_and_loading),
        ("MLflow Artifact Structure Test", test_mlflow_artifact_structure),
        ("Full Pipeline Test", run_full_pipeline_test)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 30)
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    logger.info(f"\nOverall Result: {overall_status}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)