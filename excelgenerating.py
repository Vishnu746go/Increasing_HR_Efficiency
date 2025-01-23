import pandas as pd
import numpy as np
import random

def generate_sample_data(num_employees=10):
    np.random.seed(42)  # for reproducibility
    
    data = []
    
    for i in range(num_employees):
        employee = {
            'EmployeeID': i + 1001,  # Starting from 1001
            'EmployeeName': f'Employee {i + 1}',  # Simple name generation
            
            # Numeric features using feature_ranges
            'Age': np.random.randint(18, 60),
            'DailyRate': np.random.randint(102, 1499),
            'DistanceFromHome': np.random.randint(1, 29),
            'Education': np.random.randint(1, 5),
            'EmployeeNumber': np.random.randint(1, 2000),
            'EnvironmentSatisfaction': np.random.randint(1, 4),
            'HourlyRate': np.random.randint(30, 100),
            'JobInvolvement': np.random.randint(1, 4),
            'JobLevel': np.random.randint(1, 5),
            'JobSatisfaction': np.random.randint(1, 4),
            'MonthlyIncome': np.random.randint(1009, 19999),
            'MonthlyRate': np.random.randint(2094, 26999),
            'NumCompaniesWorked': np.random.randint(0, 9),
            'PercentSalaryHike': np.random.randint(11, 25),
            'PerformanceRating': np.random.randint(1, 4),
            'RelationshipSatisfaction': np.random.randint(1, 4),
            'StockOptionLevel': np.random.randint(0, 3),
            'TotalWorkingYears': np.random.randint(0, 40),
            'TrainingTimesLastYear': np.random.randint(0, 6),
            'WorkLifeBalance': np.random.randint(1, 4),
            'YearsAtCompany': np.random.randint(0, 40),
            'YearsInCurrentRole': np.random.randint(0, 18),
            'YearsSinceLastPromotion': np.random.randint(0, 15),
            'YearsWithCurrManager': np.random.randint(0, 17),
            
            # Categorical features
            'BusinessTravel': random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']),
            'Department': random.choice(['Human Resources', 'Research & Development', 'Sales']),
            'EducationField': random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']),
            'Gender': random.choice(['Female', 'Male']),
            'JobRole': random.choice(['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                                    'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist',
                                    'Sales Executive', 'Sales Representative']),
            'MaritalStatus': random.choice(['Divorced', 'Married', 'Single']),
            'OverTime': random.choice(['No', 'Yes'])
        }
        data.append(employee)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data for 10 employees
    df = generate_sample_data(10)
    
    # Save to Excel file
    output_file = 'sample_employee_data.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Sample data has been generated and saved to {output_file}")