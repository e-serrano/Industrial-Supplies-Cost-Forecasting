# Forecasting Industrial Supplies Costs

## Project Overview
This project aims to forecast the cost of various industrial goods using historical data from the company's database. By leveraging machine learning models, the goal is to enhance procurement decision-making and optimize inventory planning.
The models used in this project are: Linear Regression, ARIMA, SES, SARIMAX and Random Forest

## Workflow
1. **Data Acquisition**: Extract relevant data from the company's database.
2. **Data Cleaning & Feature Engineering**: Use Google Colab to preprocess data, handle missing values, engineer features, and prepare datasets.
3. **Model Selection & Training**: Use multiple machine learning models and predict with each one.
4. **Pipeline Development**: Structure the Google Colab project into Python modules for data preprocessing, feature engineering, model training and evaluation.
5. **Deployment**: Deploy the final model using Streamlit for an interactive user interface. 

## Project Structure
```
├── data
│   ├── raw                     # Raw data files
│   ├── processed               # Processed data files
│
├── models                      # Trained models and references for standarization
│
├── notebooks                   # Jupyter/Colab notebooks
│
├── reports                     # Documents with results and analysis of each model
│
├── src                         # Source code files
│   ├── data_preprocessing.py   # Data cleaning and transformation
│   ├── feature_engineering.py  # Feature creation and selection
│   ├── main_app.py             # Streamlit application for deployment
│   ├── model_training_evaluation.py  # Model training and evaluation
│   ├── model_comparison.py     # Comparing different models
│
├── .gitignore                  # Git ignore file
├── LICENSE                     # Project license
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
```

## Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/e-serrano/Industrial-Supples-Cost-Forecasting.git
cd forecasting-industrial-costs
pip install -r requirements.txt
```

## Usage
Run the Streamlit app:
```sh
streamlit run main_app.py
```
Or visit the Streamlit Cloud App:
[Industrial Supplies Forecasting](https://industrial-supplies-forecasting.streamlit.app/)

## Dependencies
See `requirements.txt` for a full list of required libraries.

## License
This project is licensed under the MIT License.

## Author
Enrique Serrano García

