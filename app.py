import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('logistic_regression_model.pkl')
    categorical_encodings = joblib.load('categorical_encodings.pkl')
    training_features = joblib.load('training_features.pkl')
    return model, categorical_encodings, training_features

feature_ranges = {
    'Age': (18, 60),
    'DailyRate': (102, 1499),
    'DistanceFromHome': (1, 29),
    'Education': (1, 5),
    'EmployeeNumber': (1, 2000),
    'EnvironmentSatisfaction': (1, 4),
    'HourlyRate': (30, 100),
    'JobInvolvement': (1, 4),
    'JobLevel': (1, 5),
    'JobSatisfaction': (1, 4),
    'MonthlyIncome': (1009, 19999),
    'MonthlyRate': (2094, 26999),
    'NumCompaniesWorked': (0, 9),
    'PercentSalaryHike': (11, 25),
    'PerformanceRating': (1, 4),
    'RelationshipSatisfaction': (1, 4),
    'StockOptionLevel': (0, 3),
    'TotalWorkingYears': (0, 40),
    'TrainingTimesLastYear': (0, 6),
    'WorkLifeBalance': (1, 4),
    'YearsAtCompany': (0, 40),
    'YearsInCurrentRole': (0, 18),
    'YearsSinceLastPromotion': (0, 15),
    'YearsWithCurrManager': (0, 17)
}

categorical_options = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist',
                'Sales Executive', 'Sales Representative'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes']
}

def create_encoded_features(input_data, categorical_encodings):
    encoded_features = {}
    
    encoded_features[f'BusinessTravel_Travel_Frequently'] = 1 if input_data['BusinessTravel'] == 'Travel_Frequently' else 0
    encoded_features[f'BusinessTravel_Travel_Rarely'] = 1 if input_data['BusinessTravel'] == 'Travel_Rarely' else 0
    
    encoded_features[f'Department_Research & Development'] = 1 if input_data['Department'] == 'Research & Development' else 0
    encoded_features[f'Department_Sales'] = 1 if input_data['Department'] == 'Sales' else 0
    
    education_fields = ['Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']
    for field in education_fields:
        encoded_features[f'EducationField_{field}'] = 1 if input_data['EducationField'] == field else 0
    
    encoded_features['Gender_Male'] = 1 if input_data['Gender'] == 'Male' else 0
    
    job_roles = ['Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director',
                 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']
    for role in job_roles:
        encoded_features[f'JobRole_{role}'] = 1 if input_data['JobRole'] == role else 0
    
    encoded_features['MaritalStatus_Married'] = 1 if input_data['MaritalStatus'] == 'Married' else 0
    encoded_features['MaritalStatus_Single'] = 1 if input_data['MaritalStatus'] == 'Single' else 0
    
    encoded_features['OverTime'] = 1 if input_data['OverTime'] == 'Yes' else 0
    
    return encoded_features

def preprocess_input(user_input, categorical_encodings, training_features):
    encoded_features = create_encoded_features(user_input, categorical_encodings)
    input_data = {**user_input, **encoded_features}
    final_input = pd.DataFrame(columns=training_features)
    for feature in training_features:
        if feature in input_data:
            final_input.loc[0, feature] = input_data[feature]
        else:
            final_input.loc[0, feature] = 0
    return final_input

def validate_excel_data(df, required_features):
    missing_features = [feat for feat in required_features if feat not in df.columns]
    return missing_features

def process_excel_file(df, model, categorical_encodings, training_features):
    results = []
    
    for idx, row in df.iterrows():
        employee_data = row.to_dict()
        processed_input = preprocess_input(employee_data, categorical_encodings, training_features)
        prediction = model.predict(processed_input)[0]
        
        results.append({
            'Employee ID': row['EmployeeID'],
            'Employee Name': row['EmployeeName'],
            'Attrition Risk': 'Yes' if prediction == 1 else 'No'
        })
    
    return pd.DataFrame(results)

def bulk_prediction_tab():
    st.header("Employee Attrition Prediction")
    
    model, categorical_encodings, training_features = load_model()
    required_features = list(feature_ranges.keys()) + list(categorical_options.keys())
    
    st.write("Upload an Excel file containing employee data. Required columns: EmployeeID, EmployeeName, and the following features:")
    st.write(", ".join(required_features))
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            missing_features = validate_excel_data(df, required_features)
            
            if missing_features:
                st.error(f"Missing features in Excel file: {', '.join(missing_features)}")
                st.write("Please include all required features and try again.")
            else:
                results_df = process_excel_file(df, model, categorical_encodings, training_features)
                st.subheader("Prediction Results")
                st.dataframe(results_df)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="attrition_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

def individual_assessment_tab():
    st.header("Individual Employee Assessment")
    
    try:
        model, categorical_encodings, training_features = load_model()
        
        col1, col2, col3, col4 = st.columns(4)
        
        user_input = {}
        
        with col1:
            st.subheader("Personal Information")
            user_input['Age'] = st.slider("Age", min_value=feature_ranges['Age'][0], 
                                        max_value=feature_ranges['Age'][1], value=30)
            user_input['Gender'] = st.selectbox("Gender", options=categorical_options['Gender'],
                                              index=categorical_options['Gender'].index('Female'))
            user_input['MaritalStatus'] = st.selectbox("MaritalStatus", 
                                                     options=categorical_options['MaritalStatus'],
                                                     index=categorical_options['MaritalStatus'].index('Single'))
            user_input['Education'] = st.slider("Education", min_value=feature_ranges['Education'][0],
                                              max_value=feature_ranges['Education'][1], value=2)
            user_input['EducationField'] = st.selectbox("EducationField", 
                                                      options=categorical_options['EducationField'],
                                                      index=categorical_options['EducationField'].index('Marketing'))
            user_input['DistanceFromHome'] = st.slider("DistanceFromHome", 
                                                     min_value=feature_ranges['DistanceFromHome'][0],
                                                     max_value=feature_ranges['DistanceFromHome'][1], value=20)
            user_input['EmployeeNumber'] = st.slider("EmployeeNumber", 
                                                   min_value=feature_ranges['EmployeeNumber'][0],
                                                   max_value=feature_ranges['EmployeeNumber'][1], value=102)
        
        with col2:
            st.subheader("Job Details")
            user_input['Department'] = st.selectbox("Department", 
                                                  options=categorical_options['Department'],
                                                  index=categorical_options['Department'].index('Sales'))
            user_input['JobRole'] = st.selectbox("JobRole", options=categorical_options['JobRole'],
                                               index=categorical_options['JobRole'].index('Sales Executive'))
            user_input['JobLevel'] = st.slider("JobLevel", min_value=feature_ranges['JobLevel'][0],
                                             max_value=feature_ranges['JobLevel'][1], value=1)
            user_input['JobInvolvement'] = st.slider("JobInvolvement", 
                                                   min_value=feature_ranges['JobInvolvement'][0],
                                                   max_value=feature_ranges['JobInvolvement'][1], value=2)
            user_input['JobSatisfaction'] = st.slider("JobSatisfaction", 
                                                    min_value=feature_ranges['JobSatisfaction'][0],
                                                    max_value=feature_ranges['JobSatisfaction'][1], value=1)
            user_input['PerformanceRating'] = st.slider("PerformanceRating", 
                                                      min_value=feature_ranges['PerformanceRating'][0],
                                                      max_value=feature_ranges['PerformanceRating'][1], value=2)
            user_input['StockOptionLevel'] = st.slider("StockOptionLevel", 
                                                     min_value=feature_ranges['StockOptionLevel'][0],
                                                     max_value=feature_ranges['StockOptionLevel'][1], value=0)
        
        with col3:
            st.subheader("Compensation")
            user_input['DailyRate'] = st.slider("DailyRate", min_value=feature_ranges['DailyRate'][0],
                                              max_value=feature_ranges['DailyRate'][1], value=500)
            user_input['HourlyRate'] = st.slider("HourlyRate", 
                                               min_value=feature_ranges['HourlyRate'][0],
                                               max_value=feature_ranges['HourlyRate'][1], value=40)
            user_input['MonthlyIncome'] = st.slider("MonthlyIncome", 
                                                  min_value=feature_ranges['MonthlyIncome'][0],
                                                  max_value=feature_ranges['MonthlyIncome'][1], value=3000)
            user_input['MonthlyRate'] = st.slider("MonthlyRate", 
                                                min_value=feature_ranges['MonthlyRate'][0],
                                                max_value=feature_ranges['MonthlyRate'][1], value=15000)
            user_input['PercentSalaryHike'] = st.slider("PercentSalaryHike", 
                                                      min_value=feature_ranges['PercentSalaryHike'][0],
                                                      max_value=feature_ranges['PercentSalaryHike'][1], value=10)
        
        with col4:
            st.subheader("Work Experience & Environment")   
            user_input['BusinessTravel'] = st.selectbox("BusinessTravel", 
                                                      options=categorical_options['BusinessTravel'],
                                                      index=categorical_options['BusinessTravel'].index('Travel_Frequently'))
            user_input['OverTime'] = st.selectbox("OverTime", options=categorical_options['OverTime'],
                                                index=categorical_options['OverTime'].index('Yes'))
            user_input['NumCompaniesWorked'] = st.slider("NumCompaniesWorked", 
                                                       min_value=feature_ranges['NumCompaniesWorked'][0],
                                                       max_value=feature_ranges['NumCompaniesWorked'][1], value=4)
            user_input['TotalWorkingYears'] = st.slider("TotalWorkingYears", 
                                                      min_value=feature_ranges['TotalWorkingYears'][0],
                                                      max_value=feature_ranges['TotalWorkingYears'][1], value=5)
            user_input['YearsAtCompany'] = st.slider("YearsAtCompany", 
                                                   min_value=feature_ranges['YearsAtCompany'][0],
                                                   max_value=feature_ranges['YearsAtCompany'][1], value=1)
            user_input['YearsInCurrentRole'] = st.slider("YearsInCurrentRole", 
                                                       min_value=feature_ranges['YearsInCurrentRole'][0],
                                                       max_value=feature_ranges['YearsInCurrentRole'][1], value=0)
            user_input['YearsSinceLastPromotion'] = st.slider("YearsSinceLastPromotion", 
                                                            min_value=feature_ranges['YearsSinceLastPromotion'][0],
                                                            max_value=feature_ranges['YearsSinceLastPromotion'][1], value=0)
            user_input['YearsWithCurrManager'] = st.slider("YearsWithCurrManager", 
                                                         min_value=feature_ranges['YearsWithCurrManager'][0],
                                                         max_value=feature_ranges['YearsWithCurrManager'][1], value=0)
            user_input['WorkLifeBalance'] = st.slider("WorkLifeBalance", 
                                                    min_value=feature_ranges['WorkLifeBalance'][0],
                                                    max_value=feature_ranges['WorkLifeBalance'][1], value=1)
            user_input['EnvironmentSatisfaction'] = st.slider("EnvironmentSatisfaction", 
                                                           min_value=feature_ranges['EnvironmentSatisfaction'][0],
                                                           max_value=feature_ranges['EnvironmentSatisfaction'][1], value=1)
            user_input['RelationshipSatisfaction'] = st.slider("RelationshipSatisfaction", 
                                                            min_value=feature_ranges['RelationshipSatisfaction'][0],
                                                            max_value=feature_ranges['RelationshipSatisfaction'][1], value=1)
            user_input['TrainingTimesLastYear'] = st.slider("TrainingTimesLastYear", 
                                                          min_value=feature_ranges['TrainingTimesLastYear'][0],
                                                          max_value=feature_ranges['TrainingTimesLastYear'][1], value=0)
        
        if st.button("Predict Attrition"):
            processed_input = preprocess_input(user_input, categorical_encodings, training_features)
            prediction = model.predict(processed_input)[0]
    
            # Store the user input and prediction in session state
            st.session_state['user_input'] = user_input
            st.session_state['prediction'] = prediction
    
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("Employee is likely to leave")
            else:
                st.success("Employee is likely to stay")

    except Exception as e:
        st.error(f"Error occurred: {e}")
# Add this function to calculate SHAP values
def calculate_shap_values(model, processed_input, training_features):
    # Get the logistic regression model from the pipeline
    logistic_model = model.named_steps['logisticregression']
    
    # Create a kernel explainer instead of tree explainer
    background = pd.DataFrame(0, index=np.arange(1), columns=training_features)
    explainer = shap.KernelExplainer(logistic_model.predict_proba, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(processed_input)
    
    # For binary classification, shap_values is a list with values for each class
    # We want the values for class 1 (attrition)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create a dictionary of feature importances
    feature_importance = {}
    for idx, feature in enumerate(training_features):
        feature_importance[feature] = abs(shap_values[0][idx])
    
    return shap_values, feature_importance

def create_feature_impact_plot(feature_importance, user_input):
    # Sort features by absolute importance
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
    
    # Take top 10 features
    top_features = list(sorted_features.keys())[:10]
    importance_values = [sorted_features[f] for f in top_features]
    
    # Create more readable feature names
    feature_names = [f.replace('_', ' ').title() for f in top_features]
    
    # Create feature values text
    feature_values = []
    for feature in top_features:
        if feature in user_input:
            value = user_input[feature]
            feature_values.append(f"{value}")
        else:
            # For encoded features, check if they're 1 in the importance dict
            feature_values.append("Yes" if sorted_features[feature] > 0 else "No")

    # Create a horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feature_names,
        x=importance_values,
        orientation='h',
        text=feature_values,
        textposition='auto',
    ))

    fig.update_layout(
        title="Top 10 Factors Influencing Prediction",
        xaxis_title="Impact on Prediction",
        yaxis_title="Features",
        height=500
    )
    
    return fig

def create_recommendation_text(feature_importance, user_input, prediction):
    # Sort features by absolute importance
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))
    top_features = list(sorted_features.keys())[:5]
    
    recommendations = []
    
    if prediction == 1:  # High attrition risk
        for feature in top_features:
            if feature in feature_ranges:
                current_value = user_input[feature]
                feature_name = feature.replace('_', ' ').title()
                
                if feature in ['MonthlyIncome', 'DailyRate', 'HourlyRate']:
                    if sorted_features[feature] > 0:
                        recommendations.append(f"Consider reviewing {feature_name} (currently {current_value})")
                elif feature in ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance']:
                    if current_value < 3:
                        recommendations.append(f"Look into improving {feature_name} (currently {current_value}/4)")
            elif feature.startswith('JobRole_') and sorted_features[feature] > 0:
                recommendations.append("Consider role rotation or career development opportunities")
            elif feature == 'OverTime' and sorted_features[feature] > 0:
                recommendations.append("Review overtime policies and workload distribution")
    
    return recommendations

def explanation_tab():
    st.header("Prediction Explanation")
    
    if 'user_input' not in st.session_state:
        st.warning("Please make a prediction in the Individual Assessment tab first.")
        return
    
    try:
        model, categorical_encodings, training_features = load_model()
        user_input = st.session_state['user_input']
        
        # Process input and make prediction
        processed_input = preprocess_input(user_input, categorical_encodings, training_features)
        prediction = model.predict(processed_input)[0]
        
        # Calculate feature importance using model coefficients
        try:
            # Get the logistic regression coefficients
            logistic_model = model.named_steps['logisticregression']
            coefficients = logistic_model.coef_[0]
            
            # Calculate feature importance:
            # 1. Multiply each coefficient by its feature value to get the actual impact
            # 2. Take absolute value since we care about magnitude of impact, not direction
            feature_importance = {}
            for idx, feature in enumerate(training_features):
                feature_value = processed_input[feature].iloc[0]
                importance = abs(coefficients[idx] * feature_value)
                feature_importance[feature] = float(importance)
            
            # Normalize importance values to a 0-1 scale for easier interpretation
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                feature_importance = {
                    k: v / max_importance 
                    for k, v in feature_importance.items()
                }
            
        except Exception as importance_error:
            st.error(f"Error calculating feature importance: {importance_error}")
            return
        
        # Display prediction and probability
        st.subheader("Prediction Overview")
        if prediction == 1:
            st.error("Employee is likely to leave")
            probability = model.predict_proba(processed_input)[0][1]
            st.write(f"Probability of leaving: {probability:.2%}")
        else:
            st.success("Employee is likely to stay")
            probability = model.predict_proba(processed_input)[0][0]
            st.write(f"Probability of staying: {probability:.2%}")
        
        # Display feature importance visualization
        st.subheader("Feature Impact Analysis")
        
        try:
            # Create feature importance plot
            fig = go.Figure()
            
            # Sort features by absolute importance
            sorted_features = dict(sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            # Take top 10 most important features
            top_features = list(sorted_features.items())[:10]
            
            # Create horizontal bar chart
            fig.add_trace(go.Bar(
                y=[f[0].replace('_', ' ').title() for f in top_features],
                x=[f[1] for f in top_features],
                orientation='h',
                text=[f"{f[1]:.3f}" for f in top_features],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Top 10 Features Influencing the Prediction",
                xaxis_title="Relative Importance (0-1 scale)",
                yaxis_title="Features",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature values table
            st.subheader("Current Feature Values")
            feature_values = []
            for feature, importance in top_features:
                if feature in user_input:
                    value = user_input[feature]
                    feature_values.append({
                        "Feature": feature.replace('_', ' ').title(),
                        "Current Value": value,
                        "Relative Importance": f"{importance:.3f}"
                    })
            
            if feature_values:
                st.table(pd.DataFrame(feature_values))
            
            # Display recommendations based on the analysis
            recommendations = create_recommendation_text(feature_importance, user_input, prediction)
            if recommendations:
                st.subheader("Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")
            
        except Exception as plot_error:
            st.error(f"Error generating visualization: {plot_error}")
        
    except Exception as e:
        st.error(f"Error in generating explanation: {e}")

def main():
    tab1, tab2, tab3 = st.tabs(["Attrition Prediction", "Individual Assessment", "Prediction Explanation"])
    
    with tab1:
        bulk_prediction_tab()
    
    with tab2:
        individual_assessment_tab()
        
    with tab3:
        explanation_tab()

if __name__ == "__main__":
    main()