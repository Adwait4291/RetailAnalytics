# app.py - Streamlit app for Retail ML Pipeline (MLflow-only version with Batch Prediction)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import os
from datetime import datetime
import json
import joblib

# --- Import core functions from single_prediction.py ---
from single_prediction import (
    get_active_model_artifacts_from_mlflow,
    create_feature_df,
    process_single_record,
    make_single_prediction,
    connect_to_mongodb
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Retail ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MLflow Configuration ---
# Set this environment variable to point to your MLflow tracking server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mnt/c/Users/hp/Desktop/Retail-App-Analytics/src/models/mlruns")
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_status = "✅ Connected"
except Exception:
    mlflow_status = "❌ Disconnected"

# --- Model Configuration ---
MODEL_NAME = "Retail24RandomForestClassifier"

# --- Helper Functions ---
def safe_format_float(value, decimals=4):
    """Safely format a value to a float string, returning 'N/A' on failure."""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

# --- MLflow Utility Functions ---
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_production_model_info_from_mlflow(model_name):
    """
    Fetch metadata and metrics for the production model from MLflow.
    Loads data from the MLflow Model Registry via the 'production' alias.
    """
    client = MlflowClient()
    try:
        prod_version = client.get_model_version_by_alias(name=model_name, alias="production")
        run_info = client.get_run(prod_version.run_id)
        
        model_info = {
            "model_version": prod_version.version,
            "model_uri": f"models:/{model_name}@production",
            "run_id": prod_version.run_id,
            "trained_at": datetime.fromtimestamp(run_info.info.start_time / 1000),
            "metrics": run_info.data.metrics,
        }
        return model_info
    except MlflowException as e:
        st.error(f"Could not find a 'production' model for '{model_name}'. Please promote a model in MLflow.")
        st.error(f"MLflow error: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to fetch model info. Error: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_all_models_history(model_name):
    """Fetch the history of all versions for a given model from MLflow."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        history = []
        for v in versions:
            run_info = client.get_run(v.run_id)
            history.append({
                "model_version": v.version,
                "trained_at": datetime.fromtimestamp(run_info.info.start_time / 1000),
                "aliases": list(v.aliases),
                "metrics": run_info.data.metrics,
            })
        if not history:
            return None
        return pd.DataFrame(history).sort_values("trained_at", ascending=False)
    except MlflowException:
        return None

# --- UI Layout ---
st.title("📊 Retail ML Dashboard")
st.write("Real-time monitoring and inference for the retail machine learning pipeline.")

is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"
env_icon, env_text = ("🐳", "Running in Docker") if is_docker else ("💻", "Running Locally")

col1, col2 = st.columns(2)
col1.info(f"{env_icon} {env_text}", icon=env_icon)
col2.info(f"MLflow Status: {mlflow_status} @ {mlflow.get_tracking_uri()}", icon="🔗")

with st.spinner("Connecting to MLflow and fetching active model..."):
    active_model_info = get_production_model_info_from_mlflow(MODEL_NAME)

if active_model_info:
    # --- Sidebar ---
    st.sidebar.title("Pipeline Info")
    st.sidebar.success("Successfully connected to MLflow.")
    st.sidebar.header("Active Model")
    st.sidebar.code(f"{MODEL_NAME} - v{active_model_info['model_version']}")
    
    st.sidebar.header("Dashboard Info")
    st.sidebar.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # --- Main Content Tabs ---
    tab_metrics, tab_single, tab_batch = st.tabs(["📈 Model Metrics", "🧪 Single Prediction", "📂 Batch Prediction"])

    # --- Model Metrics Tab ---
    with tab_metrics:
        st.header("Active Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Model Version**: {active_model_info['model_version']}")
            trained_at = active_model_info['trained_at']
            st.info(f"**Trained At**: {trained_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            metrics = active_model_info.get('metrics', {})
            st.info(f"**Test Accuracy**: {safe_format_float(metrics.get('test_accuracy'))}")
            st.info(f"**Test F1 Score**: {safe_format_float(metrics.get('test_f1_score'))}")

        with col3:
            st.info(f"**Test Precision**: {safe_format_float(metrics.get('test_precision'))}")
            st.info(f"**Test Recall**: {safe_format_float(metrics.get('test_recall'))}")

        # The Feature Importance and Confusion Matrix sections have been removed.

        st.header("Model Version History")
        model_history_df = get_all_models_history(MODEL_NAME)
        if model_history_df is not None:
            model_history_df['accuracy'] = model_history_df['metrics'].apply(lambda x: x.get('test_accuracy', None))
            model_history_df['f1_score'] = model_history_df['metrics'].apply(lambda x: x.get('test_f1_score', None))
            model_history_df['trained_at_str'] = pd.to_datetime(model_history_df['trained_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(model_history_df[['model_version', 'aliases', 'trained_at_str', 'accuracy', 'f1_score']], use_container_width=True)
            
            plot_df = model_history_df.dropna(subset=['accuracy', 'f1_score']).sort_values('trained_at')
            if len(plot_df) > 1:
                st.subheader("Performance Over Time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df['trained_at'], y=plot_df['accuracy'], name="Accuracy", mode='lines+markers'))
                fig.add_trace(go.Scatter(x=plot_df['trained_at'], y=plot_df['f1_score'], name="F1 Score", mode='lines+markers'))
                fig.update_layout(title="Model Metrics Across Versions", xaxis_title="Training Time", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

    # --- Single Prediction Tab ---
    with tab_single:
        st.header("Make a Single Prediction")
        st.write("Enter feature values to get a prediction for a single user session.")
        
        with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("User & Session Info")
                platform = st.selectbox("Platform", ["iOS", "Android"])
                age = st.number_input("Age", 13, 100, 25)
                region = st.selectbox("Region", ["NorthAmerica", "Europe", "Asia", "LatinAmerica", "Oceania", "Africa", "MiddleEast"])
                user_segment = st.selectbox("User Segment", ["Young Professional", "Student", "Parent", "Senior", "Teen"])
                acquisition_channel = st.selectbox("Acquisition Channel", ["Organic", "Paid", "Referral", "Social", "Email"])
                app_version = st.text_input("App Version", "3.2.1")
                first_visit_date = st.date_input("First Visit Date", value=datetime.now())

            with col2:
                st.subheader("Engagement Metrics")
                session_count = st.number_input("Session Count", 1, 500, 10)
                total_screens_viewed = st.number_input("Total Screens Viewed", 1, 1000, 20)
                used_search_feature = 1 if st.checkbox("Used Search Feature", True) else 0
                wrote_review = 1 if st.checkbox("Wrote Review", False) else 0
                added_to_wishlist = 1 if st.checkbox("Added to Wishlist", True) else 0
                screen_list_str = st.text_area("Screen List (comma-separated)", "ProductList,ProductDetail,ShoppingCart,Checkout")

            submitted = st.form_submit_button("Predict Purchase")

        if submitted:
            sample_features = {
                "platform": platform, "age": age, "session_count": session_count,
                "total_screens_viewed": total_screens_viewed, "used_search_feature": used_search_feature,
                "wrote_review": wrote_review, "added_to_wishlist": added_to_wishlist,
                "screen_list": screen_list_str, "region": region,
                "acquisition_channel": acquisition_channel, "user_segment": user_segment,
                "app_version": app_version,
                "first_visit_date": first_visit_date.strftime('%Y-%m-%d')
            }

            with st.spinner("Loading model and making prediction..."):
                result = make_single_prediction(sample_features)
                
                if result.get("success"):
                    st.subheader("Prediction Result")
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        if result['prediction'] == 1:
                            st.success("Prediction: **Will Purchase**")
                        else:
                            st.error("Prediction: **Will NOT Purchase**")
                        st.metric("Purchase Probability", f"{result['probability']:.2%}")
                    
                    with res_col2:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=result['probability'],
                            title={'text': "Probability Gauge"},
                            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "darkblue"},
                                   'steps': [{'range': [0, 0.5], 'color': "lightgray"}, {'range': [0.5, 1], 'color': "green"}],
                                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}}
                        ))
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Feature Influence (Top 5)")
                    fi_df = pd.DataFrame(result['feature_influences']['top_features'])
                    fig = px.bar(fi_df, x='influence', y='feature', orientation='h', title="Top Influencing Features")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    db, client = connect_to_mongodb()
                    if db is not None:
                        db.predicted_results.insert_one(
                            {**sample_features, **{"purchase_24h_prediction": result['prediction'], "purchase_24h_probability": result['probability']}}
                        )
                        st.success("Prediction saved to MongoDB.")
                        if client:
                            client.close()
                else:
                    st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

    # --- Batch Prediction Tab ---
    with tab_batch:
        st.header("Make Batch Predictions")
        st.info("Upload a CSV or Excel file with user data to get predictions in bulk.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())

                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing batch predictions... This may take a moment."):
                        model, scaler, feature_names, scaling_info = get_active_model_artifacts_from_mlflow()

                        if model and scaler and feature_names and scaling_info:
                            processed_data = process_single_record(df, feature_names, scaling_info, scaler)
                            
                            if processed_data is not None:
                                predictions = model.predict(processed_data)
                                probabilities = model.predict_proba(processed_data)[:, 1]
                                
                                df['purchase_prediction'] = predictions
                                df['purchase_probability'] = probabilities
                                
                                st.success("Batch prediction complete!")
                                st.dataframe(df)
                                
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Predictions as CSV",
                                    data=csv,
                                    file_name="batch_predictions.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.error("Failed to load model artifacts for batch prediction.")
            except Exception as e:
                st.error(f"Failed to process file: {e}")

else:
    st.header("Welcome to the Retail ML Dashboard")
    st.warning("Could not connect to MLflow or find a production model. Please check your MLflow server and model registry configuration.")
    st.sidebar.error("MLflow connection failed.")