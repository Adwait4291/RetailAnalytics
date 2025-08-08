# Retail User Purchase Prediction System

**Goal:** Predict which retail mobile app users are most likely to make a purchase within the next 24 hours — enabling targeted marketing campaigns and boosting conversion rates.

## 📌 Project Highlights

* **End-to-End ML Pipeline** — Automated ingestion, preprocessing, feature engineering, and model training, implemented in modular Python scripts.
* **High-Recall Model** — Random Forest Classifier tuned for **100% recall** to ensure no potential purchasing users are missed.
* **Robust MLOps** — MLflow used for experiment tracking, model versioning, and lifecycle management.
* **Containerized Deployment** — Entire system Dockerized, including:
   * FastAPI prediction microservice
   * Streamlit monitoring dashboard

## 🔍 Technical Deep Dive

### 1️⃣ Data Processing & Feature Engineering

**What:**
* Automated pipeline ingests raw CSV user interaction logs.
* Cleans, transforms, and stores processed data in MongoDB for scalable access.

**Why:**
* MongoDB (NoSQL) handles large, semi-structured log data efficiently.
* Feature engineering converts raw behavioral logs into meaningful model features.

**How:**
* `processing.py` generates features such as:
   * **Time-based metrics** (e.g., last activity gap)
   * **Behavioral flags** (e.g., `cart_count`)
   * **Purchase intent score** (weighted heavily on cart events)

### 2️⃣ Model Training & Optimization

**What:**
* **Random Forest Classifier** for purchase prediction.

**Why:**
* Handles non-linear relationships in behavioral data.
* Target metric: **100% recall** for positive class (purchasers).

**How:**
* **GridSearchCV** tunes `n_estimators` and `max_depth`.
* **MLflow** logs:
   * Hyperparameters
   * Metrics (F1-score: `0.91`, Recall: `1.0`, ROC AUC)
   * Serialized model for version control

### 3️⃣ MLOps & Deployment

**What:**
* Full ML lifecycle orchestration, retraining triggers, and monitoring.

**Why:**
* Ensures reproducibility, scalability, and maintainability in production.

**How:**
* **Containerization**:
   * Separate Dockerfiles for pipeline & Streamlit dashboard.
   * Environment consistency across dev and prod.
* **MLflow Model Registry**:
   * Versioned models with "production" alias for live inference.
* **Automated Retraining Triggers**:
   * Data drift (+10% new data)
   * Performance degradation (-5% F1-score)
* **Deployment**:
   * FastAPI microservice for **real-time** predictions
   * Streamlit dashboard for monitoring and serving **batch** predictions

## 🛠 Tech Stack

**Data Science:**
* Pandas, NumPy, Scikit-learn

**MLOps:**
* MLflow, Docker

**Backend/API:**
* FastAPI, PyMongo

**Frontend/Dashboard:**
* Streamlit

**Database:**
* MongoDB

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/RetailAnalytics.git
cd RetailAnalytics

# Build and start services
docker compose up --build
```

Access services:
* **MLflow UI:** `http://localhost:5000`
* **API:** `http://localhost:8080/docs`
* **Streamlit Dashboard:** `http://localhost:8501`
