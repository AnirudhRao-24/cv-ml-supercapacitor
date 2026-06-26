# 🔋 ML-Based Prediction of Cyclic Voltammetry for Supercapacitors

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**🌐 Live Web Dashboard:** [https://anirudhrao-24.github.io/cv-ml-supercapacitor-bfo/](https://anirudhrao-24.github.io/cv-ml-supercapacitor-bfo/)

**🔌 Live ML API (Render):** [https://cv-ml-supercapacitor-bfo-i0cu.onrender.com/](https://cv-ml-supercapacitor-bfo-i0cu.onrender.com/)

## 📌 Project Overview
Traditional Cyclic Voltammetry (CV) testing for supercapacitor materials is a time-intensive and highly resource-dependent process. Physical synthesis and testing of multiple dopant variations create severe bottlenecks in next-generation battery research.

This project introduces a **full-stack Machine Learning pipeline** capable of predicting complete CV current curves based purely on physical input parameters (Potential, Scan Rate, Oxidation state, and Dopant Matrix). By replacing physical trial-and-error with high-accuracy AI, this tool accelerates the research of Zinc and Cobalt-substituted Bismuth Ferrite ($BiFeO_3$).

## 🚀 Key Results & Validation
The pipeline utilizes a Stacked Meta-Model approach, combining an Artificial Neural Network (ANN), Random Forest, and XGBoost using a RidgeCV Regressor. 

Tested on a completely unseen $60~mV/s$ validation dataset, the model achieved:
* **Accuracy ($R^2$ Score):** 99.74%
* **Error Rate (RMSE):** 0.000401
* **Specific Capacitance Prediction:** 114.84 F g⁻¹ *(vs 115.39 F g⁻¹ Experimental Lab Value — an error margin of 0.47%)*

## 📂 Repository Structure
```text
cv-ml-supercapacitor-bfo/
├── data/                  # Raw training datasets and unseen validation data
├── models/                # Serialized AI models (.h5, .pkl) and scalers
├── scripts/               # Modular Python backend for data processing and ML training
├── notebooks/             # Master Jupyter/Colab notebook with exploratory data analysis
├── docs/                  # Premium HTML/CSS/JS frontend application for live demos
├── render.yaml            # Deployment configuration for the Render platform
├── requirements.txt       # Python environment dependencies
└── README.md              # Project storefront

⚙️ How to Use This Repository
1. The Machine Learning Backend (Python)
To retrain the models or process new data, install the dependencies and run the modular scripts:

Bash
pip install -r requirements.txt
cd scripts
python preprocess.py
python train_ann.py
python train_rf_xgb.py
python train_meta_model.py
python evaluate.py

# To run the local API server:
uvicorn api:app --reload
2. The Interactive Dashboard (Web)
The dashboard is currently hosted live on GitHub Pages. You can visit it here:
**[https://anirudhrao-24.github.io/cv-ml-supercapacitor-bfo/](https://anirudhrao-24.github.io/cv-ml-supercapacitor-bfo/)**

To view or modify it locally, run a local web server:

Bash
cd docs
python -m http.server 8080

Use the Live Simulation tab to watch the Meta-Model predict current curves in real-time.

🛠️ Tech Stack
Backend ML: Python, TensorFlow (Keras), Scikit-Learn, XGBoost, Pandas, SciPy.

API Backend: FastAPI, Uvicorn, Pydantic.

Frontend UI: HTML5, CSS3 (Glassmorphism UI), Vanilla JS, Chart.js.

Authentication: Firebase v9.

Developed as a 6-week research project to bridge the gap between material science and artificial intelligence.
