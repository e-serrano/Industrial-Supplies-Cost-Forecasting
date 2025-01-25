from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from model_training_evaluation import train_evaluate_model
import joblib

"""
This script trains and evaluates different machine learning models to predict the unit value
of a product based on historical data. The best model is saved to a file.
"""

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "LightGBM": lgb.LGBMRegressor(random_state=42),
}


for model_name, model in models.items():
    model, average_metrics = train_evaluate_model(model)
    print(f"Average metrics for {model_name}: {average_metrics}\n")


# Save the best model
# In this case, the best model is the Random Forest.
# If you want to save a different model, change the model name.
best_model = models["Random Forest"]
joblib.dump(best_model, "models/best_model.pkl")