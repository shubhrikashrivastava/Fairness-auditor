# 🧠 Fairness Auditor

## 📌 Overview
Fairness Auditor is an end-to-end machine learning system that detects and mitigates bias in datasets.  
It demonstrates how class imbalance impacts model performance and how SMOTE improves fairness.

The project includes:
- ML pipeline for bias detection & correction  
- Flask backend API  
- Streamlit frontend for visualization  

---

## 🎯 Objective
- Detect dataset bias (class imbalance)
- Apply SMOTE to balance data
- Compare performance before & after
- Display results visually

---

## 🧩 System Architecture

Frontend (Streamlit)  
⬇  
Backend (Flask API)  
⬇  
ML Pipeline (Scikit-learn + SMOTE)  
⬇  
JSON Output → UI Display  

---

## 🧠 Machine Learning

### 📊 Dataset
- Breast Cancer Dataset (from Scikit-learn)
- No external dataset required

---

### ⚙️ Workflow
1. Load dataset  
2. Create artificial imbalance  
3. Train model (before SMOTE)  
4. Apply SMOTE  
5. Train model (after SMOTE)  
6. Compare results  

---

### 🤖 Model
- Logistic Regression  
- StandardScaler for normalization  

---

### 📈 Metrics
- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  
- Class Distribution  

---

### ⚖️ Key Insight

> Accuracy alone is not enough.  
> Fairness improves when the model performs well across all classes.

---

## 🌐 Backend (Flask API)

### 📌 Endpoint