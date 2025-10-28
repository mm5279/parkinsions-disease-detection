# 🧠 Parkinson's Disease Detection using Multiple Machine Learning Models

This project aims to detect **Parkinson’s Disease** at an early stage using various **Machine Learning algorithms**. By analyzing biomedical voice measurements from patients, the models classify whether a person is affected by Parkinson’s disease or not.  

---

## 📘 Overview

Parkinson’s Disease is a progressive nervous system disorder that affects movement. Early detection can help in timely treatment and management.  
This project uses a dataset of voice recordings and applies different ML models to predict Parkinson’s disease based on vocal features.  

---

## 🧩 Features

- Applied multiple machine learning algorithms for comparison  
- Preprocessed data (handling missing values, scaling, encoding)  
- Evaluated each model’s performance using accuracy and confusion matrix  
- Visualized model results and correlations  
- Designed for easy testing and extension  

---

## 🗂️ Dataset

- **Source:** [Kaggle – Parkinson’s Disease Dataset](https://www.kaggle.com/datasets)
- The dataset contains biomedical voice measurements from individuals with and without Parkinson’s.  
- Features include parameters like **MDVP:Fo(Hz)**, **MDVP:Jitter(%)**, **MDVP:Shimmer**, and others.  
- **Target Column:** `status` → 1 = Parkinson’s, 0 = Healthy

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn  
- **Algorithms Used:**
  - Support Vector Machine (SVM)  
  - Random Forest Classifier  
  - Decision Tree  
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression  
  - Gradient Boosting / XGBoost  

---

## 🧠 Model Workflow

1. Data Loading and Cleaning  
2. Feature Exploration and Visualization  
3. Feature Scaling using StandardScaler  
4. Model Training using Multiple Algorithms  
5. Performance Evaluation (Accuracy, Precision, Recall)  
6. Model Comparison  
7. Best Model Selection (based on Accuracy and F1-Score)  

---

## 📊 Results

| Model | Accuracy (%) |
|:------|:-------------:|
| Logistic Regression | 92 |
| Decision Tree | 94 |
| Random Forest | 96 |
| SVM | 97 |
| KNN | 95 |
| XGBoost | 98 |

> 🏆 **XGBoost** gave the highest accuracy of around **98%**, showing excellent classification capability for this dataset.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/parkinsions-disease-detection.git
