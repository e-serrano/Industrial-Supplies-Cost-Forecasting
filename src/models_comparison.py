from data_preprocessing import synthetic_data_creation
from model_training_evaluation import train_evaluate_linear_regression, train_evaluate_arima_ses, train_evaluate_sarimax, train_evaluate_random_forest 
import joblib

"""
This script trains and evaluates different machine learning models to predict the unit value
of a product based on historical data. 
Models are saved to a file.
"""

linear_regresion_model = train_evaluate_linear_regression()

arima_model, ses_model = train_evaluate_arima_ses()

df_purchases = synthetic_data_creation()

sarimax_model = train_evaluate_sarimax(df_purchases)

random_forest_model = train_evaluate_random_forest(df_purchases)

# Save each model
joblib.dump(linear_regresion_model, "models/linear_regresion_model.pkl")
arima_model.save("models/arima_model.pickle")
# joblib.dump(arima_model, "models/arima_model.pkl")
joblib.dump(ses_model, "models/ses_model.pkl")
# joblib.dump(sarimax_model, "models/sarimax_model.pkl")
sarimax_model.save("models/sarimax_model.pickle")
joblib.dump(random_forest_model, "models/random_forest_model.pkl")