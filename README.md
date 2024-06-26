# Predicting Customer Lifetime Value (CLV)

This project aims to predict customer lifetime value (CLV) using various machine learning and deep learning models based on transactional data from the Online Retail II dataset. CLV prediction helps businesses understand the future value that a customer will generate over their entire relationship with the company.

## Dataset

The dataset used for this project is the Online Retail II dataset, which contains transaction data from a UK-based online retail store between 01/12/2009 and 09/12/2011. It includes attributes such as InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.

## Approach

### 1. Data Preparation

- The dataset is loaded and cleaned to handle missing values and preprocess relevant features.
- Key features such as Recency (days since last purchase), Frequency (total number of purchases), and Monetary (total revenue generated) are aggregated at the customer level.

### 2. Exploratory Data Analysis (EDA)

- Initial insights into data distribution of Quantity and Price are visualized using histograms.
- Transaction trends over time are analyzed to understand seasonal variations and customer behavior.

### 3. Model Building and Evaluation

#### Machine Learning Models

- *RandomForestRegressor*: Ensemble learning method used to predict CLV based on Recency, Frequency, and Monetary features.
- *Linear Regression*: Basic linear model to establish a baseline for comparison.
- *XGBoost*: Gradient boosting framework known for its performance in regression tasks.

#### Deep Learning Model

- *Feedforward Neural Network (TensorFlow/Keras)*: A basic neural network architecture with dense layers to predict CLV. Includes model training, evaluation, and visualization of training history.

### 4. Model Comparison and Performance

- Models are evaluated using Mean Squared Error (MSE) and R-squared metrics to assess prediction accuracy.
- Feature importance analysis helps understand which factors contribute most to predicting CLV across different models.

## Files

- *data_cleaning.ipynb*: Jupyter notebook for data cleaning and preprocessing steps.
- *model_training.ipynb*: Jupyter notebook containing model training and evaluation code.
- *README.md*: Markdown file describing the project, dataset, approach, and model details.

## Usage

To replicate or extend this project:

1. Clone the repository:
2. Install dependencies:
3. Run the Jupyter notebooks (data_cleaning.ipynb and model_training.ipynb) to execute data preprocessing, model training, and evaluation steps.
4. Modify models, hyperparameters, or features as needed for your specific use case.

## Conclusion

Predicting customer lifetime value (CLV) is essential for businesses to optimize marketing strategies, customer retention efforts, and overall profitability. This project demonstrates different approaches to CLV prediction using both traditional machine learning and deep learning techniques, providing insights into model performance and feature importance.
