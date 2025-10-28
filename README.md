# 🧠 Parkinson’s Disease Detection using Machine Learning

This project uses **machine learning algorithms** to detect **Parkinson’s disease** based on various biomedical voice measurements.  
It compares the performance of multiple models including **Support Vector Machine (SVM)** with linear and RBF kernels, and **Logistic Regression**.

---

## 📋 Project Overview

Parkinson’s Disease is a neurodegenerative disorder that affects movement and speech. Early detection can help in effective management.  
This project analyzes the **UCI Parkinson’s dataset** and builds a predictive model to classify patients as:

- **1 → Parkinson’s Positive**  
- **0 → Healthy**

---

## 📁 Dataset

- **Source:** UCI Machine Learning Repository  
- **File Used:** `parkinsons.csv`  
- **Attributes:** 22 voice-based biomedical measurements such as:
  - `MDVP:Fo(Hz)` — Average vocal fundamental frequency
  - `MDVP:Jitter(%)` — Variation in frequency
  - `MDVP:Shimmer` — Variation in amplitude
  - `HNR` — Harmonic-to-noise ratio
  - `PPE`, `RPDE`, `DFA`, etc.

---

## ⚙️ Steps in the Notebook

### 1. **Import Dependencies**
Essential libraries like `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn` are imported.

### 2. **Data Loading and Exploration**
- The dataset is loaded and inspected for missing values, datatypes, and distribution.
- Statistical summaries are generated.

### 3. **Data Preprocessing**
- Features and target variables are separated.
- Data is standardized using `StandardScaler` for better model performance.

### 4. **Model Building**
Two SVM models are trained:
- **Linear Kernel SVM**
- **RBF Kernel SVM**

Additionally, **Logistic Regression** is trained for comparison.

### 5. **Model Evaluation**
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Classification Report

### 6. **Hyperparameter Tuning**
`GridSearchCV` is used to find the best combination of `C` and `gamma` for the RBF kernel SVM.

### 7. **Prediction System**
A small system is created to make predictions for a new set of feature inputs.

---

## 📊 Results Summary

| Model | Train Accuracy | Test Accuracy | Kernel / Type |
|--------|----------------|---------------|----------------|
| SVM (Linear) | ~95% | ~90% | Linear |
| SVM (RBF) | ~98% | ~93% | RBF |
| Logistic Regression | ~94% | ~88% | Linear Model |

*(Actual scores may vary slightly depending on dataset split and parameters.)*

---

## 🧪 Example Prediction

```python
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,
              0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,
              0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

input_data_reshaped = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = best_model.predict(std_data)

if prediction[0] == 0:
    print("The person does not have Parkinson’s disease.")
else:
    print("The person has Parkinson’s disease.")
