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

##  Next Step: Modeling  

Next, we’ll move to the **modeling phase** in `main.py`:
1. Split the clean dataset into train/test sets  
2. Build a baseline Logistic Regression model  
3. Evaluate with Accuracy, Recall, F1-score, ROC-AUC  
4. Compare with RandomForest and XGBoost  

---

##  Authors  
**Rania Mehria & Rym — Albert School 2025**  
Supervised Learning Project – Heart Failure Risk Prediction  

---

