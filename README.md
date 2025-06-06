
# 🎯 CO₂/N₂ Selectivity Predictor – Carbon Membrane App

This Streamlit-based web app predicts **CO₂/N₂ selectivity** using permeability data of carbon membrane polymers. It's built on a trained machine learning model and is designed for researchers, engineers, and students working on carbon capture technologies.

---

## 🚀 Features

✅ Easy-to-use interface  
✅ Upload or input new polymer membrane data  
✅ Predict CO₂/N₂ selectivity instantly  
✅ Visual results & interactive layout  
✅ Open-source & publicly accessible

---

## 🧪 Model Training

- Model: `RandomForestRegressor`
- Trained using a cleaned membrane dataset with 18 original features
- Feature engineering includes:
  - One-hot encoding of categorical columns
  - Grouping rare polymers
  - Handling missing values
- Output: `PolyMemCO2Pipeline.joblib` (saved ML pipeline)

---

## 📦 Requirements

Install dependencies locally:

```bash
pip install -r requirements.txt
