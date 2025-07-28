import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from datetime import timedelta

from data_preprocessing import importing_company_data
from feature_engineering import feature_engineering_sarimax, feature_engineering_random_forest

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_evaluate_linear_regression():
    """ This function predicts the unit value of a product using linear regression."""

# Data preprocessing and feature engineering
    processed_dataframe = importing_company_data()

    processed_dataframe["date_ordinal"] = processed_dataframe["order_date"].map(pd.Timestamp.toordinal)

    X = processed_dataframe[["date_ordinal"]]
    y = processed_dataframe["unit_value"]
    model = LinearRegression().fit(X, y)

    # Predict future values
    future_dates = pd.date_range(start=processed_dataframe["order_date"].max(), periods=12, freq="3ME")
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    future_preds = model.predict(future_ordinals)

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(processed_dataframe["order_date"], processed_dataframe["unit_value"], label="Histórico")
    plt.plot(future_dates, future_preds, "--", label="Predicción")
    plt.xlabel("Fecha")
    plt.ylabel("Valor unitario (€)")
    plt.title("Predicción de valor unitario")
    plt.legend()

    plt.savefig("reports/Prediccion_Regresión_Lineal.pdf", format="pdf")

    return model


def train_evaluate_arima_ses():
    """ This function predicts the unit value of a product using ARIMA and Simple Exponential Smoothing."""

# Data preprocessing and feature engineering
    processed_dataframe = importing_company_data()

# Transfrom data and dataframe
    processed_dataframe["order_date"] = pd.to_datetime(processed_dataframe["order_date"], dayfirst=True)
    processed_dataframe.set_index("order_date", inplace=True)
    processed_dataframe = processed_dataframe.sort_index()

# Reindex to monthly frequency and fill with interpolation
    df_monthly = processed_dataframe[["unit_value"]].resample("ME").mean()
    df_monthly["unit_value"] = df_monthly["unit_value"].interpolate(method="linear")

    joblib.dump(df_monthly, "models/df_monthly.pkl")

# Simple Exponential Smoothing (SES)
    ses_model = SimpleExpSmoothing(df_monthly["unit_value"]).fit()
    ses_forecast = ses_model.forecast(12)

# ARIMA
    arima_model = ARIMA(df_monthly["unit_value"], order=(1, 1, 1)).fit()
    arima_forecast = arima_model.forecast(12)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly.index, df_monthly["unit_value"], label="Datos históricos", marker='o')
    plt.plot(ses_forecast.index, ses_forecast, label="SES Forecast", linestyle="--")
    plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast", linestyle="--")
    plt.title("Forecasting de Unit Value con SES y ARIMA")
    plt.xlabel("Fecha")
    plt.ylabel("Valor Unitario (€)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig("reports/Prediccion_ARIMA_SES.pdf", format="pdf")

    return arima_model, ses_model


def train_evaluate_sarimax(df_purchases):
    """
    This function trains and evaluates a SARIMAX model on the provided data.
    Args:
        df_purchases (DataFrame): DataFrame with the initial data
    """

    df_sarimax, daily_data, target_var, exog_vars = feature_engineering_sarimax(df_purchases)

    y = daily_data[target_var]
    exog = daily_data[exog_vars]
    seasonal_period = None

    # Auto-detect seasonality if not specified
    if seasonal_period is None:
        # Search dominant periodicity (7, 30, 90 typical days)
        from scipy.fft import fft
        n = len(y)
        freqs = np.fft.fftfreq(n)
        fft_vals = np.abs(fft(y - y.mean()))

        # Search dominant frequencies
        peak_idx = np.argsort(fft_vals[1:n//2])[-3:]  # Top 3 frequencies
        periods = [int(1/freqs[i+1]) for i in peak_idx if freqs[i+1] > 0]
        periods = [p for p in periods if 7 <= p <= 365]  # Filter logic ranges

        seasonal_period = periods[0] if periods else 7
        print(f"Periodo estacional detectado: {seasonal_period} días")

    # SARIMAX setups to different volatility levels
    configs = [
        {'order': (1, 1, 1), 'seasonal_order': (0, 0, 0, 0)},
        {'order': (1, 1, 1), 'seasonal_order': (1, 0, 1, seasonal_period)},
        {'order': (2, 1, 2), 'seasonal_order': (1, 0, 1, seasonal_period)},
    ]

    best_model = None
    best_aic = np.inf

    for config in configs:
        try:
            model = SARIMAX(
                y,
                exog=exog,
                **config,
                trend='ct',  # Constant + trend
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=100)

            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_model = fitted_model
                best_config = config

        except Exception as e:
            print(f"Error con configuración {config}: {e}")
            continue

    if best_model is None:
        raise ValueError("No se pudo ajustar ningún modelo SARIMAX")

    print(f"Mejor modelo: {best_config}, AIC: {best_aic:.2f}")

# Generate diagnostics for the best model
    residuals = best_model.resid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Residuals vs time
    axes[0,0].plot(residuals)
    axes[0,0].set_title('Residuos vs Tiempo')
    axes[0,0].set_xlabel('Tiempo')
    axes[0,0].set_ylabel('Residuos')

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot de Residuos')

    # Residuals Histogram
    axes[1,0].hist(residuals, bins=30, density=True, alpha=0.7)
    axes[1,0].set_title('Distribución de Residuos')
    axes[1,0].set_xlabel('Residuos')
    axes[1,0].set_ylabel('Densidad')

    # Residuals ACF
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, ax=axes[1,1], lags=20)
    axes[1,1].set_title('ACF de Residuos')

    plt.tight_layout()
    plt.savefig("reports/Análisis_SARIMAX.pdf", format="pdf")

# Generate forecast and graph
    forecast_steps = 30
    exog_forecast = np.tile(daily_data[exog_vars].iloc[-1].values, (forecast_steps, 1))

    forecast = best_model.get_forecast(steps=forecast_steps, exog=exog_forecast)
    forecast_df = pd.DataFrame({
        'forecast': forecast.predicted_mean,
        'lower_ci': forecast.conf_int().iloc[:, 0],
        'upper_ci': forecast.conf_int().iloc[:, 1]
    })

    # Index with dates for forecast
    last_date = daily_data.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq='D'
    )
    forecast_df.index = forecast_dates
    
    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Análisis Completo: Datos Reales vs Sintéticos vs Pronóstico SARIMAX',
                fontsize=16, fontweight='bold')

    # ===== Graph 1: Historical Comparison =====
    ax1 = axes[0, 0]

    # Original data
    sample_size = min(1000, len(df_purchases))
    sample_data = df_purchases.sample(n=sample_size, random_state=42).sort_values('order_date')

    ax1.scatter(sample_data['order_date'], sample_data['unit_value'],
                alpha=0.4, s=20, color='red', label='Datos Reales (muestra)', zorder=1)

    # Daily aggregate
    ax1.plot(daily_data.index,
            daily_data['unit_value'] if 'unit_value' in daily_data.columns
            else daily_data.iloc[:, 0],
            color='green', linewidth=2, alpha=0.8, label='Datos Sintéticos (agregados)')

    ax1.set_title('Comparación Histórica: Reales vs Sintéticos', fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor Unitario')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== Graph 2: Forecasting With Intervals =====
    ax2 = axes[0, 1]

    # Recent history (last 6 months)
    recent_history = daily_data.tail(180)
    target_col = 'unit_value' if 'unit_value' in recent_history.columns else recent_history.columns[0]

    ax2.plot(recent_history.index, recent_history[target_col],
            color='blue', linewidth=2, label='Historia Reciente', alpha=0.8)

    # Forecasting
    ax2.plot(forecast_df.index, forecast_df['forecast'],
            color='orange', linewidth=3, label='Pronóstico SARIMAX')

    # Confidence intervals
    ax2.fill_between(forecast_df.index,
                    forecast_df['lower_ci'],
                    forecast_df['upper_ci'],
                    alpha=0.3, color='orange', label='Intervalo de Confianza 95%')

    ax2.set_title('Pronóstico SARIMAX con Intervalos de Confianza', fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Valor Unitario')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ===== Graph 3: Model Residuals =====
    ax3 = axes[1, 0]

    residuals = best_model.resid
    ax3.scatter(range(len(residuals)), residuals, alpha=0.6, s=20, color='purple')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax3.set_title('Residuos del Modelo SARIMAX', fontweight='bold')
    ax3.set_xlabel('Observación')
    ax3.set_ylabel('Residuo')
    ax3.grid(True, alpha=0.3)

    # Residuals statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    ax3.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ===== Graph 4: Volatility Distribution =====
    ax4 = axes[1, 1]

    # Calculate volatility rolling
    rolling_window = 30
    if len(daily_data) > rolling_window:
        rolling_std = daily_data[target_col].rolling(window=rolling_window).std()

        ax4.plot(rolling_std.index, rolling_std,
                color='darkred', linewidth=2, label=f'Volatilidad Rolling ({rolling_window}d)')
        ax4.fill_between(rolling_std.index, 0, rolling_std, alpha=0.3, color='darkred')

        # Average volatility line
        avg_volatility = rolling_std.mean()
        ax4.axhline(y=avg_volatility, color='black', linestyle='--',
                    label=f'Volatilidad Promedio: {avg_volatility:.4f}')

        ax4.set_title('Evolución de la Volatilidad', fontweight='bold')
        ax4.set_xlabel('Fecha')
        ax4.set_ylabel('Desviación Estándar Rolling')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Datos insuficientes\npara calcular volatilidad',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Volatilidad - Datos Insuficientes', fontweight='bold')

    plt.tight_layout()
    plt.savefig("reports/Predicción_SARIMAX.pdf", format="pdf")

    return best_model


def tune_hyperparameters(X_train, y_train, cv=5, conservative=False, anti_overfitting=False):
    """
    Optimize hyperparameters of Random Forest
    """
    if anti_overfitting:
        # Grid ultra-conservador para datos sintéticos
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [4, 6, 8, 10],
            'min_samples_split': [10, 15, 20, 25],
            'min_samples_leaf': [5, 8, 12, 15],
            'max_features': [0.3, 0.4, 0.5, 'sqrt']
        }
    elif conservative:
        # Grid más conservador para evitar overfitting
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [8, 12, 16, 20],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6],
            'max_features': ['sqrt', 'log2', 0.5]
        }
    else:
        # Grid original (más agresivo)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def train_evaluate_random_forest(df_purchases):
    X, df_prepared, feature_names, label_encoders = feature_engineering_random_forest(df_purchases)
    y = df_prepared['unit_value']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Fine tuning
    best_params = tune_hyperparameters(X_train, y_train, conservative=False, anti_overfitting=True)
    print(f"Mejores parámetros: {best_params}")

    # Train final mod
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }

    train_r2 = metrics['train']['r2']
    test_r2 = metrics['test']['r2']
    overfitting_gap = train_r2 - test_r2

    print(f"\n--- Análisis de Overfitting ---")
    print(f"R² Train: {train_r2:.4f}")
    print(f"R² Test:  {test_r2:.4f}")
    print(f"Gap:      {overfitting_gap:.4f}")

    if overfitting_gap > 0.1:
        print("⚠️  POSIBLE OVERFITTING detectado!")
        print("Considera usar parámetros más conservadores:")
        print("- max_depth: 10-20")
        print("- min_samples_leaf: 2-5")
        print("- min_samples_split: 5-10")
    elif overfitting_gap < 0.02:
        print("✅ Sin signos de overfitting")
    else:
        print("ℹ️  Overfitting leve, dentro del rango normal")

# Plot results of model
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Predicts vs Real Values
    axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Valores Reales')
    axes[0, 0].set_ylabel('Predicciones')
    axes[0, 0].set_title('Predicciones vs Valores Reales')

    # 2. Residuals
    residuals = y_test - y_pred_test
    axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicciones')
    axes[0, 1].set_ylabel('Residuos')
    axes[0, 1].set_title('Gráfico de Residuos')

    # 3. Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = importance_df.head(5)
    axes[1, 0].barh(top_features['feature'], top_features['importance'])
    axes[1, 0].set_xlabel('Importancia')
    axes[1, 0].set_title('Top 5 Features Más Importantes')

    # 4. Residuals Distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuos')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Residuos')

    plt.tight_layout()
    plt.savefig("reports/Análisis_Random_Forest.pdf", format="pdf")

# Forecasting future values
    last_date = df_purchases['order_date'].max()
    days_ahead=360
    delivery_days=[7, 14, 21, 28]
    suppliers=['DELCORTE', 'SIDSA', 'EGARENSE']

    if model is None:
        raise ValueError("El modelo debe ser entrenado primero")

    # Generate future dates
    all_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')

    weekday_dates = all_dates[all_dates.dayofweek < 5]

    # Add aleatory gaps
    np.random.seed(42)
    has_order = np.random.random(len(weekday_dates)) > 0.4 # Only 60% of workdays have orders

    future_dates = weekday_dates[has_order]

    # Create forecasting DataFrame
    n_predictions = len(future_dates)
    future_df = pd.DataFrame({
        'order_date': future_dates,
        'delivery_date': future_dates + pd.to_timedelta(np.random.choice(delivery_days, size=n_predictions), unit='D'),
        'quantity': np.random.choice(range(50,500,10), size=n_predictions),
        'supplier_name': np.random.choice(suppliers, size=n_predictions)
    })

    # Prepare features
    X_future, _, _, _, = feature_engineering_random_forest(future_df, label_encoders, df_prepared['order_date'].min())

    # Predictions
    predictions = model.predict(X_future)

    future_df['predicted_unit_value'] = predictions

# Graphing the predictions
    # Sorting dates
    df_real_sorted = df_purchases.sort_values('order_date')
    df_forecast_sorted = future_df.sort_values('order_date')

    plt.figure(figsize=(12, 6))

    # Real data
    plt.plot(df_real_sorted['order_date'], df_real_sorted['unit_value'],
            label='Real', color='blue', marker='o', markersize=4, linewidth=1)

    # Forecasting data
    plt.plot(df_forecast_sorted['order_date'], df_forecast_sorted['predicted_unit_value'],
            label='Forecast', color='orange', linestyle='--', marker='x', markersize=6, linewidth=0.5)

    # Graph setup
    plt.title('Evolución del unit_value (Real vs Forecast)')
    plt.xlabel('Fecha')
    plt.ylabel('Unit Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("reports/Predicción_Random_Forest.pdf", format="pdf")

    return model