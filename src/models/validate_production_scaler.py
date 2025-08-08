import os
import joblib
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Replace these if needed
MLFLOW_TRACKING_URI = "file:///mnt/c/Users/hp/Desktop/Retail-App-Analytics/src/models/mlruns"
MODEL_NAME = "Retail24RandomForestClassifier"
PRODUCTION_ALIAS = "production"

def validate_scaler_object(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        if not hasattr(scaler, "transform"):
            print(f"❌ Invalid scaler object: {type(scaler)}. Missing .transform().")
            return False
        print(f"✅ Valid scaler object loaded: {type(scaler)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
        return False

def main():
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    client = MlflowClient()

    try:
        # Get the production model version
        prod_version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
        run_id = prod_version.run_id
        print(f"ℹ️  Production model version: {prod_version.version}, run ID: {run_id}")

        # Download the artifacts of the production model
        local_dir = client.download_artifacts(run_id=run_id, path="preprocessor")
        scaler_path = os.path.join(local_dir, "preprocessor.pkl")

        print(f"🔍 Validating scaler at: {scaler_path}")
        is_valid = validate_scaler_object(scaler_path)

        if is_valid:
            print("✅ Scaler validation PASSED.")
        else:
            print("❌ Scaler validation FAILED. Consider retraining or fixing the model artifacts.")
    
    except MlflowException as e:
        print(f"❌ MLflow Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    main()
