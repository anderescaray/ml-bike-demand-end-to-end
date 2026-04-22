# 🚴‍♂️ ML Bike Demand Prediction (End-to-End MLOps)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Kedro](https://img.shields.io/badge/kedro-1.3.1-black.svg)
![uv](https://img.shields.io/badge/uv-fast_dependency_management-magenta)
![Docker](https://img.shields.io/badge/docker-containerized-blue)
![Dash](https://img.shields.io/badge/UI-Dash-008DB8)

An end-to-end Machine Learning project to predict bike rental demand. This project covers the entire ML lifecycle—from data ingestion and processing to model training, evaluation, and serving—all wrapped in a professional and reproducible MLOps pipeline.

## 📌 Project Overview

This repository demonstrates best practices in MLOps and production-ready machine learning code. It is designed to be highly modular, testable, and reproducible. 

The pipeline predicts the number of bikes rented per hour based on weather and calendar information, treating it as a regression problem.

## 🏗️ Architecture & Tech Stack

*   **Dependency Management**: [uv](https://github.com/astral-sh/uv) for blazing-fast python environment management.
*   **Orchestration & Data Catalog**: [Kedro](https://kedro.org/) for building robust, scalable, and reproducible data pipelines.
*   **Machine Learning**: Scikit-Learn & CatBoost for high-performance gradient boosting regression.
*   **Containerization**: Docker & Docker Compose to ensure consistent environments across local development and production.
*   **Frontend / UI**: A web-based interactive dashboard built with [Plotly Dash](https://dash.plotly.com/).
*   **Linting & Formatting**: Ruff for extremely fast Python linting and code formatting.

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

# Install the project and its dependencies
uv pip install -e .
```

## 🏃‍♂️ Running the Pipelines

Data processing and model training are orchestrated using Kedro.

To execute the entire pipeline (data processing -> feature engineering -> model training -> evaluation):
```bash
kedro run
```

*Tip: You can also run specific sub-pipelines or nodes using Kedro's standard CLI (e.g., `kedro run --pipeline=data_processing`).*

## 🐳 Running with Docker

For a fully isolated and reproducible execution, the project is containerized. 

To build the image and spin up the services (which will train the model and start the Dash UI):
```bash
docker-compose up --build
```
*Note: Make sure your Docker daemon is running before executing this command.*

## 📊 Dashboard UI

Once the model is trained and the dashboard service is running (either locally or via Docker), you can interact with the model predictions via the Dash frontend.

Navigate to: `http://localhost:8050` *(or the port defined in your configuration)*.

---

## 👏 Acknowledgements

This project was developed as part of the MLOps Bootcamp by **[Timur Bikmukhametov]**. A huge thanks for the fantastic content, architecture best practices, and guidance!

🔗 **His LinkedIn profile:** [https://www.linkedin.com/in/timurbikmukhametov/]
