#!/usr/bin/env python3
"""
SARIMAX Time Series Prediction for Application and Project Type Analytics
UPAHO Zoning Management System

This script implements SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
for predicting future application trends based on project types.
"""

import sys
import json
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mysql.connector
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_db_connection():
    """Get database connection from config.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'u520834156_dbUPAHOZoning'
        }
    
    try:
        conn = mysql.connector.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            charset='utf8mb4'
        )
        return conn
    except Exception as e:
        print(json.dumps({'success': False, 'error': f'Database connection failed: {str(e)}'}))
        sys.exit(1)

def load_time_series_data(project_type_id=None, metric_type='application_count', months_back=24):
    """Load time series data from database"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Build query
        query = """
            SELECT 
                date,
                year,
                month,
                project_type_id,
                application_count,
                approved_count,
                rejected_count,
                total_area_sqm,
                avg_cost,
                approval_rate
            FROM application_time_series
            WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s MONTH)
        """
        
        params = [months_back]
        
        if project_type_id:
            query += " AND project_type_id = %s"
            params.append(project_type_id)
        
        query += " ORDER BY date ASC"
        
        cursor.execute(query, params)
        data = cursor.fetchall()
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Select the metric
        metric_map = {
            'application_count': 'application_count',
            'approved_count': 'approved_count',
            'total_area': 'total_area_sqm',
            'avg_cost': 'avg_cost',
            'approval_rate': 'approval_rate'
        }
        
        if metric_type not in metric_map:
            metric_type = 'application_count'
        
        series = df[metric_map[metric_type]].fillna(0)
        
        return series, df
    
    except Exception as e:
        print(json.dumps({'success': False, 'error': f'Error loading data: {str(e)}'}))
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

def load_exogenous_variables(start_date, end_date):
    """Load exogenous variables for SARIMAX"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = """
            SELECT date, variable_name, variable_value
            FROM exogenous_variables
            WHERE date BETWEEN %s AND %s
            ORDER BY date ASC
        """
        
        cursor.execute(query, [start_date, end_date])
        data = cursor.fetchall()
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot to get variables as columns
        exog_df = df.pivot(index='date', columns='variable_name', values='variable_value')
        exog_df = exog_df.ffill().bfill()
        
        return exog_df
    
    except Exception as e:
        # Return None if exogenous variables not available
        return None
    finally:
        cursor.close()
        conn.close()

def check_stationarity(series):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    return result[1] <= 0.05  # p-value <= 0.05 means stationary

def auto_arima(series, seasonal=True, max_p=3, max_d=2, max_q=3, max_P=2, max_D=1, max_Q=2, m=12):
    """Automatically find best SARIMA parameters using auto_arima"""
    try:
        model = pm.auto_arima(
            series,
            seasonal=seasonal,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            max_P=max_P,
            max_D=max_D,
            max_Q=max_Q,
            m=m,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        return model.order, model.seasonal_order, model.aic()
    except Exception as e:
        # Fallback to simple ARIMA
        return (1, 1, 1), (1, 1, 1, 12), None

def train_sarimax_model(series, exog=None, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Train SARIMAX model"""
    try:
        if exog is not None and len(exog) == len(series):
            model = SARIMAX(series, exog=exog, order=order, seasonal_order=seasonal_order)
        else:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        
        fitted_model = model.fit(disp=False, maxiter=200)
        return fitted_model
    except Exception as e:
        print(json.dumps({'success': False, 'error': f'Model training failed: {str(e)}'}))
        sys.exit(1)

def generate_forecast(model, steps, exog_future=None):
    """Generate forecast using trained model"""
    try:
        forecast = model.forecast(steps=steps, exog=exog_future)
        conf_int = model.get_forecast(steps=steps, exog=exog_future).conf_int()
        
        return forecast, conf_int
    except Exception as e:
        print(json.dumps({'success': False, 'error': f'Forecast generation failed: {str(e)}'}))
        sys.exit(1)

def save_prediction_to_db(project_type_id, metric_type, forecast_dates, predictions, lower_bounds, upper_bounds, model_order, model_metrics):
    """Save predictions to database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Save predictions
        insert_query = """
            INSERT INTO sarimax_predictions 
            (project_type_id, prediction_date, forecast_date, predicted_value, lower_bound, upper_bound, 
             metric_type, model_parameters, sarimax_order, aic_score, mse, mae, rmse)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            predicted_value = VALUES(predicted_value),
            lower_bound = VALUES(lower_bound),
            upper_bound = VALUES(upper_bound),
            updated_at = CURRENT_TIMESTAMP
        """
        
        prediction_date = datetime.now().date()
        
        for i, forecast_date in enumerate(forecast_dates):
            cursor.execute(insert_query, (
                project_type_id,
                prediction_date,
                forecast_date,
                float(predictions[i]),
                float(lower_bounds[i]) if lower_bounds is not None else None,
                float(upper_bounds[i]) if upper_bounds is not None else None,
                metric_type,
                json.dumps(model_metrics),
                str(model_order),
                model_metrics.get('aic'),
                model_metrics.get('mse'),
                model_metrics.get('mae'),
                model_metrics.get('rmse')
            ))
        
        # Update model performance
        perf_query = """
            INSERT INTO sarimax_model_performance
            (project_type_id, metric_type, model_order, aic_score, mse, mae, rmse, 
             training_data_points, forecast_horizon, is_active, model_parameters, last_trained)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s, NOW())
            ON DUPLICATE KEY UPDATE
            aic_score = VALUES(aic_score),
            mse = VALUES(mse),
            mae = VALUES(mae),
            rmse = VALUES(rmse),
            last_trained = VALUES(last_trained),
            updated_at = CURRENT_TIMESTAMP
        """
        
        cursor.execute(perf_query, (
            project_type_id,
            metric_type,
            str(model_order),
            model_metrics.get('aic'),
            model_metrics.get('mse'),
            model_metrics.get('mae'),
            model_metrics.get('rmse'),
            model_metrics.get('training_data_points'),
            len(forecast_dates),
            json.dumps(model_metrics)
        ))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        print(json.dumps({'success': False, 'error': f'Error saving predictions: {str(e)}'}))
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function to run SARIMAX prediction"""
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No input file provided'}))
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        with open(input_file, 'r') as f:
            params = json.load(f)
    except Exception as e:
        print(json.dumps({'success': False, 'error': f'Error reading input file: {str(e)}'}))
        sys.exit(1)
    
    project_type_id = params.get('project_type_id')
    metric_type = params.get('metric_type', 'application_count')
    forecast_months = params.get('forecast_months', 12)
    months_back = params.get('months_back', 24)
    auto_select = params.get('auto_select_order', True)
    
    # Load time series data
    result = load_time_series_data(project_type_id, metric_type, months_back)
    if result is None:
        print(json.dumps({'success': False, 'error': 'No time series data available'}))
        sys.exit(1)
    
    series, df = result
    
    if len(series) < 12:
        print(json.dumps({'success': False, 'error': 'Insufficient data points (need at least 12 months)'}))
        sys.exit(1)
    
    # Load exogenous variables if available
    start_date = series.index[0]
    end_date = series.index[-1]
    exog = load_exogenous_variables(start_date, end_date)
    
    # Auto-select order if requested
    if auto_select:
        order, seasonal_order, aic = auto_arima(series)
    else:
        order = tuple(params.get('order', [1, 1, 1]))
        seasonal_order = tuple(params.get('seasonal_order', [1, 1, 1, 12]))
    
    # Train model
    model = train_sarimax_model(series, exog, order, seasonal_order)
    
    # Prepare future exogenous variables if available
    exog_future = None
    if exog is not None:
        # Generate future dates
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_months, freq='MS')
        
        # Extend exogenous variables (simple forward fill for now)
        exog_future = exog.reindex(future_dates).ffill()
    
    # Generate forecast
    forecast, conf_int = generate_forecast(model, forecast_months, exog_future)
    
    # Calculate model metrics
    fitted_values = model.fittedvalues
    residuals = model.resid
    
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(mse)
    
    model_metrics = {
        'aic': float(model.aic),
        'bic': float(model.bic),
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'training_data_points': len(series),
        'order': order,
        'seasonal_order': seasonal_order
    }
    
    # Generate forecast dates
    last_date = series.index[-1]
    forecast_dates = [(last_date + pd.DateOffset(months=i+1)).date() for i in range(forecast_months)]
    
    # Prepare historical data
    historical = []
    for date, value in series.items():
        historical.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': float(value)
        })
    
    # Prepare forecast data
    forecast_data = []
    for i, date in enumerate(forecast_dates):
        forecast_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_value': float(forecast.iloc[i]),
            'lower_bound': float(conf_int.iloc[i, 0]) if conf_int is not None else None,
            'upper_bound': float(conf_int.iloc[i, 1]) if conf_int is not None else None
        })
    
    # Save to database if project_type_id is provided
    if project_type_id:
        try:
            save_prediction_to_db(
                project_type_id,
                metric_type,
                forecast_dates,
                forecast.values,
                conf_int.iloc[:, 0].values if conf_int is not None else None,
                conf_int.iloc[:, 1].values if conf_int is not None else None,
                f"{order}{seasonal_order}",
                model_metrics
            )
        except Exception as e:
            # Continue even if save fails
            pass
    
    # Return results
    result = {
        'success': True,
        'project_type_id': project_type_id,
        'metric_type': metric_type,
        'model_metrics': model_metrics,
        'historical': historical,
        'forecast': forecast_data,
        'model_order': f"{order}{seasonal_order}"
    }
    
    print(json.dumps(result, default=str))
    
    # Clean up input file
    try:
        os.remove(input_file)
    except:
        pass

if __name__ == '__main__':
    main()

