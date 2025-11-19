#  Heart Failure Diagnosis – Machine Learning Project  


##  Business Challenge  
Heart failure is a leading cause of hospitalization and death in Europe.  
It generates high healthcare costs and puts heavy pressure on hospital resources.  

**Objective:**  
Build a supervised machine learning model that predicts the risk of heart failure based on patient data (age, cholesterol, blood pressure, etc.).  

**Business Value:**  
- Improve preventive care and early diagnosis  
- Reduce hospital readmissions and costs  
- Support data-driven decisions for doctors and insurers  

---

## Dataset Description  

**Source:**  
[Heart Failure Diagnosis Dataset – Kaggle](https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning)

**Overview:**  
- Records: **70,000 patients**  
- Features: **13**  
- Target variable: `cardio` (1 = heart disease, 0 = healthy)

| Feature | Description | Type |
|----------|--------------|------|
| age | Patient age in days | Numerical |
| gender | Gender (1 = female, 2 = male) | Categorical |
| height | Height in cm | Numerical |
| weight | Weight in kg | Numerical |
| ap_hi | Systolic blood pressure | Numerical |
| ap_lo | Diastolic blood pressure | Numerical |
| cholesterol | Cholesterol level (1–3) | Ordinal |
| gluc | Glucose level (1–3) | Ordinal |
| smoke | Smoking habit (0 = no, 1 = yes) | Binary |
| alco | Alcohol consumption (0 = no, 1 = yes) | Binary |
| active | Physical activity (0 = no, 1 = yes) | Binary |
| cardio | Presence of heart disease | Target |

---

##  Exploratory Data Analysis (EDA)  

**Steps completed:**  
1. **Data Loading & Structure:**  
   - Dataset loaded successfully (`70,000 rows × 13 columns`)  
   - No missing or duplicated data 
   - All features are numeric  

2. **Descriptive Statistics:**  
   - Mean, min, and max show realistic distributions  
   - Some extreme values in `ap_hi` (blood pressure) indicate outliers  

3. **Target Balance:**  
   - `cardio` variable is balanced → 50% sick / 50% healthy  

4. **Correlations:**  
   - `age`, `cholesterol`, and `gluc` show moderate correlation with `cardio`  
   - No multicollinearity → all features kept  

5. **Visualizations:**  
   - Histograms for `age`, `height`, `weight`  
   - Boxplots comparing patients with/without heart disease  
   - Heatmap for numeric correlations  

6. **Key Insights:**  
   - Older patients and those with higher blood pressure are more likely to develop heart disease  
   - Cholesterol and glucose levels also contribute to risk  

7. **Clean Dataset Saved:**  
   - File: `data/heart_failure_clean.csv`  
   - Ready for modeling phase   

---

##  Modeling

This section describes the machine learning models tested and the selection of the final model.

### 1. Logistic Regression (Baseline Model)

Logistic Regression was used as the benchmark due to its simplicity and interpretability.

**Performance:**
- Accuracy: ~0.71  
- Recall: ~0.67  
- ROC-AUC: ~0.71  

This baseline provides a reference level of performance.  
However, it does not capture non-linear patterns present in medical data.

---

### 2. Random Forest Classifier

Random Forest was selected due to its ability to:
- Handle nonlinear relationships  
- Manage outliers  
- Work well with tabular medical datasets  

**Default Performance:**
- Accuracy: ~0.72  
- Recall: ~0.69  
- ROC-AUC: ~0.72  

The model already outperforms Logistic Regression.

#### Hyperparameter Tuning  
GridSearchCV and RandomizedSearchCV were used to optimize:
- `n_estimators`  
- `max_depth`  
- `min_samples_split`  
- `min_samples_leaf`  

**Optimized Random Forest – Final Performance:**
- Accuracy: 0.733  
- Recall: 0.683  
- Precision: 0.758  
- F1-score: 0.718  
- ROC-AUC: 0.732  

This is the best performing model overall.

---

### 3. Support Vector Machine (SVM)

SVM with an RBF kernel was evaluated for comparison.

**Performance:**
- Accuracy: 0.725  
- Recall: 0.694  
- ROC-AUC: 0.782  

Although it achieves an excellent ROC-AUC, the training time was significantly higher (×30), making it less suitable for large-scale or real-time deployment.

---

### 4. Final Model Selection

The final selected model is the **Optimized Random Forest Classifier**.

**Reasons for selection:**
- Best balance between accuracy, recall, and F1-score  
- More robust to noise and measurement variability  
- Scales efficiently to large datasets  
- Provides feature importance, which is useful for clinical interpretation  

**Most influential features:**
1. Systolic blood pressure (`ap_hi`)  
2. Diastolic blood pressure (`ap_lo`)  
3. Age  
4. Cholesterol  
5. Weight  

These variables are medically consistent with known risk factors for heart disease.

---

##  Business Impact of the Final Model

The optimized Random Forest model supports hospitals and healthcare professionals by:

- Improving early detection of heart disease  
- Supporting preventive care and reducing late-stage diagnosis  
- Reducing hospital readmissions and associated costs  
- Assisting clinicians with data-driven insights  
- Prioritizing high-risk patients for follow-up examinations  

The model is not intended to replace medical judgment but to assist clinicians in making faster and more informed decisions.

---
##  How to Run the Project

This section explains how to execute the project step-by-step.

---

### 1. Clone the Repository

```bash
git clone <URL_OF_THE_REPOSITORY>
cd heart-failure-ml
````

---

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
```

**Activate the environment:**

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

---

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Jupyter Notebook (EDA + Modeling)

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open the file:

```
eda.ipynb
```

This notebook contains all steps:

* dataset exploration
* preprocessing
* data visualization
* model training
* hyperparameter tuning
* final model selection
* model export (`model.pkl`)

---

### 5. Load and Use the Final Model 
The finalized optimized **Random Forest model** is saved as:

```
model.pkl
```

Example of how to load and run a prediction:

```python
import joblib
import numpy as np

bundle = joblib.load("model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]

# Example input (13 features in correct order)
sample = np.array([[50, 1, 160, 70, 120, 80, 2, 1, 0, 0, 1]])

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(prediction)
```

---

### 6. Project Structure

```
heart-failure-ml/
│
├── data/
│   ├── heart_failure.csv
│   └── heart_failure_clean.csv
│
├── eda.ipynb
├── main.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

### 7. Notes

* This project does **not** include an API deployment.
* The exported model is the **Optimized Random Forest**.
* All experiments (LogReg, RF, SVM, XGBoost) are inside `eda.ipynb`.



##  Final Results & Model Summary

This section summarizes the performance of all tested models and highlights the final selected model used for deployment.

### 1. Model Comparison Summary

| Model                     | Accuracy | Recall | Precision | F1-score | ROC-AUC | Training Time |
|--------------------------|----------|--------|-----------|----------|---------|----------------|
| Logistic Regression       | ~0.71    | ~0.67  | ~0.73     | ~0.69    | 0.71    | Fast           |
| Random Forest (Default)  | ~0.72    | ~0.69  | ~0.75     | ~0.71    | 0.72    | Medium         |
| Random Forest (Optimized)| **0.733**| **0.683**| **0.758**| **0.718**| **0.732**| Medium |
| SVM (RBF Kernel)         | 0.725    | 0.694  | 0.739     | 0.716    | **0.782**| **Very Slow**  |

**Key Observations:**
- SVM achieves the **highest ROC-AUC**, but at the cost of long training time.
- Logistic Regression is reliable but underperforms on non-linear medical data.
- **Optimized Random Forest** provides the **best overall balance** between accuracy, recall, precision, and speed.

---

### 2. Selected Final Model: Optimized Random Forest

The chosen model is the **Optimized Random Forest Classifier**, selected for:

- Strong and balanced performance across all metrics  
- High robustness to noise, variability, and outliers  
- Ability to capture non-linear medical relationships  
- Interpretability via **feature importance**  
- Suitability for hospital-scale datasets (70,000 patients)  

---

### 3. Most Important Features

The model identifies the following as the strongest predictors of heart disease:

| Rank | Feature       | Medical Meaning                     |
|------|---------------|-------------------------------------|
| 1    | `ap_hi`       | Systolic blood pressure             |
| 2    | `ap_lo`       | Diastolic blood pressure            |
| 3    | `age`         | Patient age                         |
| 4    | `cholesterol` | Cholesterol level                   |
| 5    | `weight`      | Body weight                         |

These findings are **aligned with real clinical risk factors**, increasing confidence in the model's reliability.

---

### 4. Final Model Artifact

The trained and optimized Random Forest model is exported as: model.pkl 


This file includes:

- The final Random Forest model  
- The scaler used for preprocessing  
- The list of feature names  

This ensures the model can be reused or integrated into future systems without retraining.

---

### 5. Conclusion

The optimized Random Forest model offers:

- Solid predictive performance  
- Fast inference  
- Interpretability  
- Clinical consistency  

It is well-suited to support **preventive cardiology**, **risk stratification**, and **hospital decision-making** while remaining simple to run and maintain.



##  Authors  
**Rania Mehria & Ryme Belouahri — Albert School 2025**  
Supervised Learning Project – Heart Failure Risk Prediction  

---

