# ⚙️ Universal Bearing Guard AI - Predictive Maintenance System

A machine learning-powered predictive maintenance dashboard for detecting bearing failures before they occur. Trained on NASA IMS bearing run-to-failure datasets, this system uses unsupervised learning (PCA) to identify anomalies in bearing vibration patterns.

## 🎯 Overview

Traditional maintenance approaches are either **reactive** (fix after failure) or **preventive** (replace on schedule). This system enables **predictive maintenance** by monitoring bearing health in real-time and alerting users to degradation before catastrophic failure.

### Key Features

- **🔴 Real-Time Monitoring**: Live sensor simulation with immediate anomaly detection
- **📂 Flexible Data Upload**: Upload sensor data of **any shape** - AI automatically transforms it
- **📊 Live Simulation on Custom Data**: Run real-time analysis on your uploaded datasets
- **🔍 Root Cause Diagnosis**: Identify which bearing and sensor features are contributing to failures
- **🗄️ PostgreSQL Database**: Persistent storage for all predictions and analysis history
- **🌐 Modern Web UI**: React frontend with FastAPI backend

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Overview │ │  Guide   │ │   Live   │ │  Batch   │ ...       │
│  │   Tab    │ │   Tab    │ │Simulation│ │ Analysis │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                          │
│  ┌──────────────┐ ┌────────────────┐ ┌───────────────────────┐ │
│  │   /predict   │ │ /preprocess-   │ │ /simulate-custom/next │ │
│  │              │ │    upload      │ │                       │ │
│  └──────────────┘ └────────────────┘ └───────────────────────┘ │
│                              │                                  │
│  ┌────────────────────────────────────────────────────────────┐│
│  │              DataTransformer Module                        ││
│  │  Detects: raw_signal | partial_features | full_features   ││
│  │  Outputs: Standard 20-feature format (5 × 4 bearings)      ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌─────────────┐  ┌────────────┐  ┌────────────┐
      │  PCA Model  │  │ PostgreSQL │  │   Scaler   │
      │ (universal_ │  │  Database  │  │ (Standard) │
      │model.joblib)│  │            │  │            │
      └─────────────┘  └────────────┘  └────────────┘
```

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Technical Details](#-technical-details)
- [Model Information](#-model-information)

## 🚀 Quick Start

### 1. Start PostgreSQL Database
```bash
docker-compose up -d
```

### 2. Start Backend (FastAPI)
```bash
cd f:\predictive_mainteance_bearings
.\venv\Scripts\activate
uvicorn main:app --reload
```
Backend runs at: `http://localhost:8000`

### 3. Start Frontend (React)
```bash
cd frontend
npm install
npm start
```
Frontend runs at: `http://localhost:3000`

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker & Docker Compose (for PostgreSQL)

### Step 1: Clone & Setup Python Environment
```bash
cd f:\predictive_mainteance_bearings
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Setup Frontend
```bash
cd frontend
npm install
```

### Step 3: Start Database
```bash
docker-compose up -d
```

### Step 4: Train Model (if needed)
```bash
python train.py
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/model-info` | GET | Get model threshold and info |
| `/predict` | POST | Single prediction (20 features) |
| `/analyze-batch` | POST | Batch analysis (upload CSV) |
| `/preprocess-upload` | POST | **NEW**: Upload any CSV, auto-transform to 20 features |
| `/simulate/reset` | POST | Reset NASA data simulation |
| `/simulate/next` | GET | Get next NASA data point |
| `/simulate-custom/reset` | POST | Reset custom data simulation |
| `/simulate-custom/next` | GET | Get next custom data point |
| `/simulate-custom/status` | GET | Check custom simulation status |
| `/health-check` | GET | Full system diagnostics |
| `/records` | GET | Get all prediction records |
| `/history` | GET | Get last 50 predictions |

## 📁 Project Structure

```
predictive_mainteance_bearings/
├── main.py                    # FastAPI backend (all endpoints)
├── data_transformer.py        # NEW: Intelligent data format converter
├── database.py                # SQLAlchemy database connection
├── sql_models.py              # Database table definitions
├── train.py                   # Model training script
├── universal_model.joblib     # Trained PCA model + scaler + threshold
├── simulation_data.csv        # NASA IMS data for simulation
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # PostgreSQL container config
│
├── src/
│   ├── config.py              # Configuration settings
│   ├── ingestion.py           # Data loading utilities
│   └── features.py            # Feature extraction functions
│
├── frontend/
│   ├── package.json           # Node.js dependencies
│   └── src/
│       └── App.js             # React frontend application
│
└── README.md                  # This file
```

## ✨ Features

### 1. Overview Tab 🏠
- System status dashboard
- **Live threshold from trained model**
- Quick start navigation

### 2. Project Guide Tab 📚
- Predictive maintenance concepts
- Bearing failure progression timeline
- Feature explanations

### 3. Live Simulation Tab 📡
- Replay NASA IMS bearing failure data
- Adjustable simulation speed
- Real-time anomaly chart
- Progress tracking (9,464 data points)
- Alert log for critical failures

### 4. Batch Analysis Tab 📂 (ENHANCED)
**NEW: Flexible Data Upload**
- Upload **any CSV file** - no format restrictions!
- AI automatically detects input format:
  - **Raw signals**: Extracts RMS, Kurtosis, Skewness, Peak, Crest
  - **Partial features**: Pads missing columns
  - **Full features**: Direct mapping
  - **Extra columns**: Maps to 4 bearings, discards excess
- Transformation preview before simulation
- **Live simulation on your data** with real-time chart

### 5. Diagnostics Tab 🔍
- Feature importance visualization
- Identify critical sensors
- Root cause analysis

### 6. Model Info Tab 📊
- Model specifications
- Feature breakdown per bearing
- Action guidelines

## 🔬 Technical Details

### Data Transformer Module

The `data_transformer.py` module intelligently handles any input format:

```python
from data_transformer import transform_data

# Transform any DataFrame to standard 20-feature format
transformed_df, info = transform_data(any_dataframe)
print(info['detected_format'])  # 'raw_signal', 'partial_features', etc.
```

### Supported Input Formats

| Format | Description | Transformation |
|--------|-------------|----------------|
| **Raw Signal** | Time-series vibration data | Extracts 5 features per channel |
| **Partial Features** | < 20 columns | Maps + pads with zeros |
| **Full Features** | Exactly 20 columns | Direct mapping |
| **Extra Features** | > 20 columns | Maps to 4 bearings |

### Features (20 Total)

For each of 4 bearings, 5 statistical features:

| Feature | Formula | Description |
|---------|---------|-------------|
| **RMS** | √(Σx²/N) | Overall vibration energy |
| **Kurtosis** | E[(X-μ)⁴]/σ⁴ | Spikiness (early failure indicator) |
| **Skewness** | E[(X-μ)³]/σ³ | Distribution asymmetry |
| **Peak** | max(\|x\|) | Maximum amplitude |
| **Crest Factor** | Peak/RMS | Normalized impact severity |

## 🤖 Model Information

### Training Data
- **Source**: NASA IMS Center
- **Datasets**: 3 complete bearing run-to-failure experiments
- **Total Records**: 9,464 snapshots
- **Sampling Rate**: 20 kHz

### Model Specifications
| Property | Value |
|----------|-------|
| Algorithm | PCA (Principal Component Analysis) |
| Input Features | 20 (4 bearings × 5 metrics) |
| Scaler | StandardScaler |
| Anomaly Detection | Reconstruction Error (MSE) |
| Threshold | Dynamic (from training data) |

### Interpreting Results

| Anomaly Score | Status | Action |
|---------------|--------|--------|
| Below threshold | ✅ HEALTHY | Continue operation |
| Above threshold | 🚨 CRITICAL_FAILURE | Stop & inspect immediately |

## 🗄️ Database Schema

### predictions Table
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    input_features JSON,
    anomaly_score FLOAT,
    threshold FLOAT,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 📚 References

- **NASA IMS Dataset**: https://data.nasa.gov/dataset/ims-bearings
- PCA for Anomaly Detection
- FastAPI Documentation: https://fastapi.tiangolo.com
- React Documentation: https://react.dev

---

## 🔮 Future Enhancements

- [ ] Implement online learning for model adaptation
- [ ] Add IoT integration for real sensor data
- [ ] Implement email/SMS alerting

---

*This project is provided for educational and research purposes.*
