# Employee Attrition Analysis

This repository contains a comprehensive analysis of employee attrition, aiming to identify key factors contributing to employee turnover and to develop predictive models to anticipate such events. The analysis is conducted using Python, leveraging various data science and machine learning libraries.

## Project Overview

Employee attrition refers to the gradual reduction of the workforce due to resignations, retirements, or other factors. Understanding and predicting attrition is crucial for organizations to maintain operational efficiency and employee satisfaction. This project explores the IBM HR Analytics Employee Attrition & Performance dataset to:

- Analyze the factors influencing employee attrition.
- Develop predictive models to forecast which employees are likely to leave.
- Provide actionable insights to help in employee retention strategies.

## Dataset

The dataset used in this analysis is the [IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). It includes various features related to employee demographics, job roles, and performance metrics.

## Repository Contents

- `main.ipynb`: Jupyter Notebook containing the complete analysis, including data preprocessing, exploratory data analysis (EDA), model building, and evaluation.

## Key Features Analyzed

- **Age**: Employee's age.
- **Attrition**: Whether the employee has left the company.
- **BusinessTravel**: Frequency of travel for business purposes.
- **Department**: Department in which the employee works.
- **DistanceFromHome**: Distance between the employee's home and workplace.
- **EducationField**: Field of education.
- **Gender**: Employee's gender.
- **JobRole**: Role of the employee within the company.
- **MaritalStatus**: Marital status of the employee.
- **MonthlyIncome**: Monthly income of the employee.
- **OverTime**: Whether the employee works overtime.
- **TotalWorkingYears**: Total years of professional experience.
- **YearsAtCompany**: Number of years spent at the current company.
- **YearsInCurrentRole**: Number of years in the current role.
- **YearsSinceLastPromotion**: Years since the last promotion.
- **YearsWithCurrManager**: Years working with the current manager.

## Analysis and Modeling

The analysis follows these steps:

1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
2. **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships between features and attrition.
3. **Model Building**: Developing predictive models using algorithms such as Logistic Regression, Decision Trees, and Random Forests.
4. **Model Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, F1-score, ROC-AUC, and log loss.

## Results

The models developed provide insights into the factors that contribute most significantly to employee attrition. Key findings include:

- Employees with higher overtime are more likely to leave.
- Job roles and departments have varying attrition rates.
- Longer tenure with the current manager correlates with lower attrition.

## Conclusion

Understanding employee attrition through data analysis enables organizations to implement targeted strategies to improve employee retention. By identifying the key factors that influence attrition, companies can take proactive measures to enhance employee satisfaction and reduce turnover.
