# 🚴‍♂️ ML Bike Demand Prediction (End-to-End MLOps)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Kedro](https://img.shields.io/badge/kedro-1.3.1-black.svg)
![uv](https://img.shields.io/badge/uv-fast_dependency_management-magenta)
![Docker](https://img.shields.io/badge/docker-containerized-blue)
![Dash](https://img.shields.io/badge/UI-Dash-008DB8)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white)

An end-to-end Machine Learning project to predict bike rental demand. This project covers the entire ML lifecycle—from data ingestion and processing to model training, evaluation, and serving—all wrapped in a professional and reproducible MLOps pipeline.

## 📌 Project Overview

This repository demonstrates best practices in MLOps and production-ready machine learning code. It is designed to be highly modular, testable, and reproducible. 

The pipeline predicts the number of bikes rented per hour based on weather and calendar information, treating it as a regression problem.

## 🏗️ Architecture & Tech Stack

* **Dependency Management**: [uv](https://github.com/astral-sh/uv) for blazing-fast python environment management.
* **Orchestration & Data Catalog**: [Kedro](https://kedro.org/) for building robust, scalable, and reproducible data pipelines.
* **Machine Learning**: Scikit-Learn & CatBoost for high-performance gradient boosting regression.
* **Experiment Tracking**: [MLflow](https://mlflow.org/) integrated via `kedro-mlflow` to automatically log hyperparameters and evaluation metrics.
* **Model Serving (API)**: [FastAPI](https://fastapi.tiangolo.com/) providing a real-time REST endpoint for model predictions.
* **Containerization**: Docker & Docker Compose to ensure consistent environments across local development and production.
* **Frontend / UI**: A web-based interactive dashboard built with [Plotly Dash](https://dash.plotly.com/) consuming the FastAPI endpoint.
* **Testing & Linting**: `pytest` for robust unit testing and Ruff for extremely fast Python linting and formatting.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd ml-bike-demand-end-to-end
```

### 2. Environment Setup (using `uv`)
We use `uv` for modern dependency management because of its unparalleled speed and reliability.

```bash
# Create a virtual environment
uv venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Sync the project and its dependencies (reads from pyproject.toml and uv.lock)
uv sync
```

## 🏃‍♂️ Running the Pipelines

Data processing and model training are orchestrated using Kedro.

To execute the entire pipeline (data processing -> feature engineering -> model training -> evaluation):
```bash
kedro run
```

*Tip: You can also run specific sub-pipelines or nodes using Kedro's standard CLI (e.g., `kedro run --pipeline=training`).*

## 🌟 Extended Features (Beyond the Bootcamp)

While the foundation of this project comes from the original MLOps Bootcamp, the architecture has been professionalized and extended with several production-ready features:

1. **REST API for Inference**: Replaced the static batch inference loop with a **FastAPI** web server. The model is loaded into memory on startup and serves predictions in real-time via a `POST /predict` endpoint.
2. **Experiment Tracking**: Integrated **MLflow** via the `kedro-mlflow` plugin. Hyperparameters (like CatBoost config) and evaluation metrics (MAE, RMSE, MAPE) are automatically logged for every training pipeline run.
3. **Unit Testing**: Implemented a modern testing suite using **pytest** and **pytest-cov** to validate the core data processing and metric computation logic, ensuring robustness and high code coverage.

## 🐳 Running with Docker

For a fully isolated and reproducible execution, the project is containerized. 

To build the image and spin up all the microservices (Training, FastAPI Inference, Dash UI, and MLflow Dashboard):
```bash
docker-compose up --build
```
*Note: Make sure your Docker daemon is running before executing this command.*

Once everything is up and running, you will have access to the following services:
- 📊 **Dash UI**: `http://localhost:8050` (Real-time monitoring dashboard)
- 🚀 **FastAPI Docs**: `http://localhost:8000/docs` (Swagger UI for the inference endpoint)
- 📈 **MLflow Tracking**: `http://localhost:5000` (Experiment tracking dashboard)

---

## 🗺️ Roadmap / Future Work

Continuous improvement is a core principle of MLOps. The upcoming phases for this repository include:
- **CI/CD Integration**: Implementing **GitHub Actions** to automate unit testing (CI) on every push and manage continuous deployment (CD) directly to cloud infrastructure.
- **Data Drift Monitoring**: Adding tools like Evidently AI to monitor shifting data distributions over time and trigger automated retraining pipelines.

---

## 👏 Acknowledgements

This project was developed as part of the MLOps Bootcamp by **Timur Bikmukhametov**. A huge thanks for the fantastic content, architecture best practices, and guidance!

🔗 **His LinkedIn profile:** [Timur Bikmukhametov](https://www.linkedin.com/in/timurbikmukhametov/)