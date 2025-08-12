# QoSGuard 🛡️

**A lightweight, real-time network anomaly detection & QoS recommender**

QoSGuard ingests network flow telemetry, flags anomalies using ML, and suggests QoS actions (throttle/prioritize/isolate) for production networks.

## 🚀 Quick Start

```bash
# Local development
docker-compose up -d
```

Visit http://localhost:3000 for the dashboard and http://localhost:8000/docs for API docs.

## 📁 Architecture

```
qosguard/
├── app/           # FastAPI backend
├── ui/            # Next.js dashboard  
├── models/        # ML model training & registry
├── data_pipeline/ # ETL and feature engineering
├── simulator/     # Network traffic simulator
├── infra/         # CloudFormation IaC
├── tests/         # Test suites
└── docker/        # Container configs
```

## 🎯 Features

- **Real-time Detection**: ML-powered anomaly detection on network flows
- **QoS Recommendations**: Intelligent policy suggestions (PRIORITIZE, RATE_LIMIT, DROP, INSPECT)
- **Interactive Dashboard**: Live metrics, visualizations, and SHAP explanations
- **Production Ready**: Observability, testing, CI/CD, and cloud deployment
- **Easy Deploy**: One command local setup, AWS deployment with CloudFormation

## 🔧 Tech Stack

**Backend**: FastAPI, MLflow, Prometheus  
**Frontend**: Next.js, Tailwind CSS, WebSocket  
**ML**: scikit-learn, LightGBM, PyTorch, SHAP  
**Data**: UNSW-NB15 dataset  
**Infrastructure**: Docker, AWS Lambda, DynamoDB, S3

## 📊 Models & Metrics

- **Logistic Regression**: Baseline linear model
- **LightGBM**: Gradient boosting for tabular data
- **PyTorch MLP**: Neural network approach

**Evaluation**: ROC-AUC, PR-AUC, F1, Recall@FPR with class imbalance handling

## 🚀 Deployment

**Local**: `docker-compose up`  
**AWS**: `make deploy-aws` (In Progress)

## 📈 Performance Goals

- API latency: p50 ≤ 300ms (warm)
- Lambda cold start: ≤ 1.5s median
- Test coverage: ≥ 80%

---

*Built with ❤️ for production network security*
