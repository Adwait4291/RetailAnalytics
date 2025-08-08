# 🛍️ Retail User Purchase Prediction System

> **Predict which retail mobile app users are most likely to make a purchase within the next 24 hours — enabling targeted marketing campaigns and boosting conversion rates.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-orange.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Project Highlights

- **🔄 End-to-End ML Pipeline** — Automated ingestion, preprocessing, feature engineering, and model training, implemented in modular Python scripts
- **🎯 High-Recall Model** — Random Forest Classifier tuned for 100% recall to ensure no potential purchasing users are missed
- **⚙️ Robust MLOps** — MLflow used for experiment tracking, model versioning, and lifecycle management
- **🐳 Containerized Deployment** — Entire system Dockerized, including FastAPI prediction microservice and Streamlit monitoring dashboard

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Ingestion│───▶│ Feature Engine  │───▶│  Model Training │
│      (CSV)      │    │   (MongoDB)     │    │    (MLflow)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │◄───│   FastAPI       │◄───│ Model Registry  │
│   Dashboard     │    │     API         │    │   (Production)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔍 Technical Deep Dive

### 1️⃣ Data Processing & Feature Engineering

**What:**
- Automated pipeline ingests raw CSV user interaction logs
- Cleans, transforms, and stores processed data in MongoDB for scalable access

**Why:**
- MongoDB (NoSQL) handles large, semi-structured log data efficiently
- Feature engineering converts raw behavioral logs into meaningful model features

**How:**
`processing.py` generates features such as:
- **Time-based metrics** (e.g., last activity gap)
- **Behavioral flags** (e.g., cart_count)
- **Purchase intent score** (weighted heavily on cart events)

### 2️⃣ Model Training & Optimization

**What:**
- Random Forest Classifier for purchase prediction

**Why:**
- Handles non-linear relationships in behavioral data
- Target metric: 100% recall for positive class (purchasers)

**How:**
- GridSearchCV tunes `n_estimators` and `max_depth`
- MLflow logs:
  - Hyperparameters
  - Metrics (F1-score: 0.91, Recall: 1.0, ROC AUC)
  - Serialized model for version control

### 3️⃣ MLOps & Deployment

**What:**
- Full ML lifecycle orchestration, retraining triggers, and monitoring

**Why:**
- Ensures reproducibility, scalability, and maintainability in production

**How:**

**Containerization:**
- Separate Dockerfiles for pipeline & Streamlit dashboard
- Environment consistency across dev and prod

**MLflow Model Registry:**
- Versioned models with "production" alias for live inference

**Automated Retraining Triggers:**
- Data drift (+10% new data)
- Performance degradation (-5% F1-score)

**Deployment:**
- FastAPI microservice for real-time predictions
- Streamlit dashboard for monitoring and serving batch predictions

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Data Science** | Pandas, NumPy, Scikit-learn |
| **MLOps** | MLflow, Docker |
| **Backend/API** | FastAPI, PyMongo |
| **Frontend/Dashboard** | Streamlit |
| **Database** | MongoDB |

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.8+ (for local development)
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/RetailAnalytics.git
cd RetailAnalytics

# Build and start services
docker compose up --build
```

### Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **MLflow UI** | http://localhost:5000 | Experiment tracking & model registry |
| **FastAPI** | http://localhost:8080/docs | Interactive API documentation |
| **Streamlit Dashboard** | http://localhost:8501 | Monitoring & batch predictions |

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **F1-Score** | 0.91 |
| **Recall** | 1.00 |
| **Precision** | 0.84 |
| **ROC AUC** | 0.95 |

## 🔧 Project Structure

```
RetailAnalytics/
├── 📁 data/
│   ├── raw/                    # Raw CSV files
│   └── processed/              # Cleaned datasets
├── 📁 src/
│   ├── processing.py           # Feature engineering pipeline
│   ├── training.py             # Model training & evaluation
│   └── api.py                  # FastAPI prediction service
├── 📁 dashboard/
│   └── streamlit_app.py        # Monitoring dashboard
├── 📁 docker/
│   ├── Dockerfile.pipeline     # ML pipeline container
│   └── Dockerfile.dashboard    # Streamlit container
├── 📄 docker-compose.yml       # Multi-service orchestration
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md
```

## 🧪 API Usage

### Predict Single User

```python
import requests

response = requests.post(
    "http://localhost:8080/predict",
    json={
        "user_id": "12345",
        "last_activity_hours": 2.5,
        "cart_count": 3,
        "session_duration": 15.2
    }
)

print(response.json())
# Output: {"prediction": 1, "probability": 0.89}
```

### Batch Predictions

```python
response = requests.post(
    "http://localhost:8080/predict/batch",
    json={
        "users": [
            {"user_id": "12345", "cart_count": 3, ...},
            {"user_id": "67890", "cart_count": 0, ...}
        ]
    }
)
```

## 🔄 Model Retraining

The system automatically triggers retraining when:

- **Data Drift Detected**: New data volume increases by 10%
- **Performance Degradation**: F1-score drops below 0.86
- **Manual Trigger**: Via Streamlit dashboard

```python
# Manual retraining
python src/training.py --retrain --data-path data/new_data.csv
```

## 📈 Monitoring & Observability

The Streamlit dashboard provides:

- **Real-time Metrics**: Prediction accuracy, response times
- **Data Quality Checks**: Missing values, outliers, drift detection
- **Model Performance**: Confusion matrix, ROC curves
- **Business Impact**: Conversion rates, campaign effectiveness

## 🧑‍💻 Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MongoDB (if not using Docker)
mongod --dbpath /path/to/db

# Run components individually
python src/processing.py      # Data pipeline
python src/training.py        # Model training
uvicorn src.api:app --reload  # API server
streamlit run dashboard/streamlit_app.py  # Dashboard
```

### Testing

```bash
# Run unit tests
pytest tests/

# Integration tests
python tests/test_api.py

# Performance tests
python tests/test_performance.py
```

## 📋 Roadmap

- [ ] **Real-time Streaming**: Kafka/Kinesis integration for live data
- [ ] **Advanced Models**: Deep learning with TensorFlow/PyTorch
- [ ] **A/B Testing**: Integrated experimentation framework
- [ ] **Multi-cloud**: AWS/GCP deployment options
- [ ] **Explainability**: SHAP integration for model interpretability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern MLOps best practices
- Inspired by real-world retail analytics challenges
- Community feedback and contributions welcome

---

**⭐ Star this repository if you find it helpful!**

For questions or support, please [open an issue](https://github.com/<your-username>/RetailAnalytics/issues).
