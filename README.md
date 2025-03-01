
<img width="1440" alt="Ekran Resmi 2025-03-01 14 56 25" src="https://github.com/user-attachments/assets/afcdab49-2216-4dd8-be1e-9f2339d5ddce" />

# Bank Marketing Dataset Analysis and Model Deployment Using Ensemble Model

## Project Description

This project focuses on developing a predictive model using the Bank Marketing Dataset to determine which customers are more likely to subscribe to the bank's term deposit product during telemarketing campaigns. The primary objective is to leverage machine learning techniques to enhance the efficiency of customer targeting and campaign performance. The project scope includes comprehensive analysis and cleaning of the dataset, feature engineering to improve predictive power, and the removal of outliers to ensure data quality. Multiple machine learning models are evaluated and optimized to achieve the best possible performance. Finally, a user-friendly interface is designed using Streamlit to deploy the model, allowing users to easily input customer data and receive predictions in real-time.

## Dataset Basic Properties

The dataset used in this project is named bank-additional.csv, which contains information related to a bank's telemarketing campaigns. The dataset includes 4119 rows (samples) and 20 input features along with 1 target variable. The target variable, y, is a binary variable that indicates whether a customer has subscribed to the term deposit product. The possible values for the target variable are:

- "yes" – The customer subscribed to the term deposit product.
- "no" – The customer did not subscribe to the term deposit product.

This dataset serves as the foundation for building a predictive model to optimize the bank's marketing strategies.

**Removed Features**: Features such as month, day_of_week, and poutcome were removed during preprocessing due to low relevance or redundancy.

## Data Cleaning and Preprocessing

Several preprocessing steps were applied to the dataset to ensure data quality and suitability for machine learning models:

1. **Handling Missing Data**:
   - Missing values were replaced with appropriate strategies such as using the mean for numerical features or the most frequent value for binary features.
   - Entries with "unknown" or "nonexistent" values were treated as missing data and appropriately handled to avoid data bias.

2. **Encoding of Categorical Variables**:
   - Categorical features were encoded using Label Encoding, transforming each category into numerical labels suitable for machine learning algorithms.
   - Binary categorical variables, such as "yes" and "no," were mapped to 1 and 0, respectively, for compatibility.

3. **Scaling of Numerical Variables**:
   - Numerical features were standardized using the StandardScaler, ensuring all numerical features have a mean of 0 and a standard deviation of 1, which helps models interpret these features consistently.

4. **Detection and Removal of Outliers**:
   - The Z-Score Method was employed to identify and remove outliers. Data points with Z-scores outside the range of ±3 were considered outliers and excluded from the dataset.

5. **Cleaning of Duplicate Records**:
   - Duplicate rows were identified and removed to improve data quality and avoid redundancy.

This comprehensive preprocessing pipeline ensured the dataset was ready for training robust and reliable machine learning models.

## Exploratory Data Analysis (EDA)

The following analyses were performed to understand the structure of the dataset and the distribution of features:

- **Target Variable Analysis**:
  - Class Distribution: The no class was detected at a rate of 89.1% and the yes class at a rate of 10.9%.
  - Result: A serious class imbalance was observed in the dataset.

- **Feature Distributions**:
  - Numerical Features: The distributions of numerical variables were examined using histograms.
  - Categorical Features: The category distributions were analyzed with Countplot.

- **Correlation Between Features**:
  - Highly correlated variables were detected using the correlation matrix.
  - Example: 97% correlation was found between emp.var.rate and euribor3m.

- **Outlier Detection**:
  - Extreme values in numerical features were analyzed and cleaned.

## Feature Selection and Engineering

- **Recursive Feature Elimination (RFE)**:
  - The 10 most effective features were selected using RandomForestClassifier.
- **New Features**: Additional features such as quarter derived from existing data were added to the dataset.

## Model Selection and Optimization

5 different machine learning models were evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machines (SVM)
- Voting Classifier

**Hyperparameter Optimization**: The best parameters of the models were determined using GridSearchCV.

**Best Model**:
- Voting Classifier (Ensemble): Accuracy rate was increased by combining multiple models.

## Conclusion

As a result of the performance evaluation of the developed model, the ROC-AUC score was 0.97 and the accuracy value was 93%. These results show that the model works with high accuracy and provides reliable predictions for campaign strategies. Thanks to the distribution of the model via Streamlit, banks can analyze campaign processes in real time and make customer-focused marketing decisions quickly.

In this direction, the developed solution can be considered as an important tool that strengthens data-driven decision-making processes in the banking sector and increases customer acquisition rates.

## How to Use the App

1. Adjust the input parameters using the sliders and dropdown menus to match the customer profile
2. Click the "Predict" button to generate a prediction
3. View the prediction result and probability score
4. Use the "Reset All Fields" button in the sidebar to clear all inputs


## Requirements

- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Pickle
=======
# bank-prediction
>>>>>>> d3e8afdeb373d12140d2ac5ab673040e1f3427e5
