# 🌾 Lite XAI Crop Recommender

A **lightweight, mobile-friendly** web app for crop recommendation using Machine Learning (RF & XGBoost) and Explanation AI (SHAP & LIME).

## 🚀 Quick Start (Deployment Ready)

This app is optimized for instant deployment and fast loading (500-sample training).

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run App
```bash
streamlit run app.py
```

The app will start instantly and train models on the fly.

## 📱 Features
- **Fast Models**: Uses Random Forest & XGBoost.
- **Explainable AI**:
  - **SHAP**: Why the model made this prediction.
  - **LIME**: Local explanation for specific cases.
  - **Counterfactuals**: "What-if" analysis (closest alternative).
- **Mobile Optimized**: Responsive layout and touch-friendly inputs.
