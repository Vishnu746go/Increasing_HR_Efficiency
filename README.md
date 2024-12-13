# Increasing_HR_Efficiency

# Employee Attrition Prediction using Machine Learning

This repository contains the implementation of a **Machine Learning model** for predicting employee attrition using a variety of features. The model is built using the **TPOT AutoML** tool and aims to predict if an employee is likely to leave the company based on their attributes such as job satisfaction, work-life balance, salary, and more.

## Overview

The dataset used in this project contains various features about employees, including demographic information, job satisfaction, work-related details, and more. We processed the data, applied feature engineering techniques, and used **AutoML (TPOT)** to identify the best model for predicting employee attrition.

### **Key Features**
- **Data Preprocessing:** Handles missing values, scales numerical data, and encodes categorical variables.
- **AutoML:** Uses **TPOT** to automate model selection and optimization.
- **Prediction:** The model predicts employee attrition based on the input features.

---

## 📂 Files in this Repository

- **[MajorProject_implementation1.ipynb](./AttritionPrediction.ipynb.ipynb):** This is the Jupyter Notebook containing the full implementation of data loading, preprocessing, feature engineering, model training, and evaluation.
- [best_pipeline.py](./best_pipeline.py): This is the best pipeline generated by TPOT for predicting employee attrition.
- [logistic_regression_model.pkl](./logistic_regression_model.pkl): This is the saved model used for predictions after training.
- [Employee_Attrition.csv](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset):  This is the dataset used for the project

---

## 🛠️ Tools and Technologies Used

- **Programming Language:** Python
- **Libraries/Frameworks:**
  - **Pandas**: For data manipulation and analysis.
  - **Scikit-learn**: For machine learning preprocessing and modeling.
  - **TPOT**: For AutoML and model optimization.
  - **Joblib**: For saving and loading the trained model.
  - **StandardScaler**: For feature scaling.
- **Development Environment:** Jupyter Notebook

---

## 🚀 Methodology

1. **Data Loading and Preprocessing:**
   - Loaded the employee attrition dataset.
   - Handled missing values, scaled numerical data, and applied one-hot encoding to categorical variables.
   
2. **Feature Engineering:**
   - Dropped unnecessary features like 'StandardHours', 'EmployeeCount', etc.
   - Focused on important numerical features such as **WorkLifeBalance**, **YearsAtCompany**, etc.
   
3. **Model Training with AutoML (TPOT):**
   - Applied **TPOTClassifier** for automated machine learning, which automatically explored different models and hyperparameters to find the best-performing model.
   
4. **Prediction:**
   - The trained model was used to predict employee attrition. Predictions are based on various employee attributes.
   
5. **Evaluation:**
   - Evaluated the model’s performance using accuracy.

---

## 📊 Results

- The AutoML process identified **Logistic Regression** as the best model.
- The final model achieved high accuracy in predicting employee attrition, with a well-optimized pipeline.


---

## 🖼️ Sample Prediction

To see the **predicted attrition status** based on the user input:

```python
# Example input values
user_input = {
    'Age': 20,
    'DailyRate': 5000,
    'DistanceFromHome': 20,
    'Education': 3,
    'EnvironmentSatisfaction': 2,
    'BusinessTravel': 'Non-Travel',
    'Department': 'Research & Development',
    'Gender': 'Male',
    'JobRole': 'Human Resources',
    'MaritalStatus': 'Single',
    'OverTime': 'Yes'
}
The predicted attrition status is: No
```
## 📫 Contact
For inquiries or collaborations, feel free to connect:
- **GitHub:** [Vishnu746go](https://github.com/Vishnu746go)
- **LinkedIn:** [Vishnu Prava](https://www.linkedin.com/in/vishnu-prava/)
- **Email:** vishnuprava2003@gmail.com


