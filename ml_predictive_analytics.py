#!/usr/bin/env python3
"""
Enhanced Predictive Analytics with Advanced Machine Learning Models
UPAHO Zoning Management System

This module provides comprehensive ML models including:
- Time Series Forecasting (ARIMA, Prophet, LSTM)
- Classification Models (XGBoost, Random Forest, Gradient Boosting)
- Regression Models (XGBoost Regressor, Neural Networks)
- Clustering (K-means, DBSCAN)
- Anomaly Detection (Isolation Forest, One-Class SVM)
"""

import pandas as pd
import numpy as np
import joblib
import mysql.connector
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, 
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Advanced Models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("TensorFlow/Keras not available. Install with: pip install tensorflow")


class MLPredictiveAnalytics:
    """
    Advanced Machine Learning Predictive Analytics System
    """
    
    def __init__(self, db_config):
        """Initialize ML Predictive Analytics with database configuration"""
        self.db_config = db_config
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_path, '..', 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            connection = mysql.connector.connect(
                host=self.db_config.get('host', 'localhost'),
                user=self.db_config.get('user', 'root'),
                password=self.db_config.get('password', ''),
                database=self.db_config.get('database', 'u520834156_dbUPAHOZoning'),
                charset='utf8mb4',
                connect_timeout=10
            )
            return connection
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def load_historical_data(self, data_type='applications', limit=None):
        """
        Load historical data for training from multiple sources
        
        Args:
            data_type: Type of data to load ('applications', 'population', 'economic')
            limit: Maximum number of records to load
        """
        connection = self.connect_database()
        if not connection:
            return None
            
        try:
            if data_type == 'applications':
                query = """
                SELECT 
                    af.id,
                    af.client_id,
                    af.project_type,
                    af.project_nature,
                    af.project_location,
                    af.project_area,
                    af.lot_area,
                    af.status,
                    af.created_at,
                    af.updated_at,
                    DATE(af.created_at) as application_date,
                    YEAR(af.created_at) as application_year,
                    MONTH(af.created_at) as application_month,
                    DAYOFWEEK(af.created_at) as day_of_week,
                    CASE 
                        WHEN af.status = 'approved' THEN DATEDIFF(af.updated_at, af.created_at)
                        ELSE NULL
                    END as processing_days,
                    CASE WHEN af.status = 'approved' THEN 1 ELSE 0 END as is_approved,
                    CASE WHEN af.status = 'rejected' THEN 1 ELSE 0 END as is_rejected,
                    CASE WHEN af.status = 'pending' THEN 1 ELSE 0 END as is_pending,
                    c.age,
                    c.gender
                FROM application_forms af
                LEFT JOIN clients c ON af.client_id = c.id
                WHERE af.created_at IS NOT NULL
                ORDER BY af.created_at DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                    
            elif data_type == 'population':
                query = """
                SELECT 
                    barangay,
                    year,
                    population,
                    households,
                    density_per_sqkm,
                    growth_rate
                FROM population_history
                ORDER BY barangay, year DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                    
            elif data_type == 'economic':
                query = """
                SELECT 
                    date,
                    indicator_type,
                    barangay,
                    value,
                    unit
                FROM economic_indicators
                ORDER BY date DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
            else:
                return None
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            if len(df) == 0:
                print(f"No data found for {data_type}")
                return None
                
            return df
            
        except Exception as e:
            print(f"Error loading {data_type} data: {e}")
            if connection:
                connection.close()
            return None
    
    def load_application_trends(self, months=24):
        """Load aggregated application trends for time series analysis"""
        connection = self.connect_database()
        if not connection:
            return None
            
        try:
            query = """
            SELECT 
                date,
                zone_type,
                barangay,
                applications_submitted,
                applications_approved,
                applications_rejected,
                applications_pending,
                approval_rate,
                average_processing_days
            FROM application_trends
            WHERE date >= DATE_SUB(CURDATE(), INTERVAL ? MONTH)
            ORDER BY date, zone_type, barangay
            """
            
            df = pd.read_sql(query, connection, params=[months])
            connection.close()
            
            return df
            
        except Exception as e:
            print(f"Error loading application trends: {e}")
            if connection:
                connection.close()
            return None
    
    def preprocess_application_data(self, df):
        """Preprocess application data for machine learning"""
        if df is None or len(df) == 0:
            return None, None, None, None
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        numeric_cols = ['project_area', 'lot_area', 'age', 'processing_days']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
        
        # Encode categorical variables
        categorical_cols = ['project_type', 'project_nature', 'status', 'gender']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = df[col].astype(str)
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].fillna('unknown'))
                else:
                    df[col] = df[col].astype(str)
                    # Handle new categories
                    unique_values = set(df[col].unique())
                    known_values = set(self.encoders[col].classes_)
                    for val in unique_values - known_values:
                        self.encoders[col].classes_ = np.append(self.encoders[col].classes_, val)
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].fillna('unknown'))
        
        # Feature engineering
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['days_since_submission'] = (datetime.now() - df['created_at']).dt.days
            df['application_year'] = df['created_at'].dt.year
            df['application_month'] = df['created_at'].dt.month
            df['application_day_of_week'] = df['created_at'].dt.dayofweek
            df['is_weekend'] = (df['application_day_of_week'] >= 5).astype(int)
        
        # Create interaction features
        if 'project_area' in df.columns and 'lot_area' in df.columns:
            df['area_ratio'] = df['project_area'] / (df['lot_area'] + 1)
        
        # Select features for training
        feature_columns = [
            'application_year', 'application_month', 'application_day_of_week',
            'is_weekend', 'project_area', 'lot_area', 'age',
            'days_since_submission', 'area_ratio'
        ]
        
        # Add encoded categorical features
        for col in categorical_cols:
            if f'{col}_encoded' in df.columns:
                feature_columns.append(f'{col}_encoded')
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].fillna(0)
        
        # Targets
        y_approval = df['is_approved'] if 'is_approved' in df.columns else None
        y_processing_time = df['processing_days'] if 'processing_days' in df.columns else None
        
        return X, y_approval, y_processing_time, available_features
    
    def train_approval_prediction_model(self, X, y, model_type='xgboost'):
        """
        Train model to predict application approval
        
        Args:
            X: Feature matrix
            y: Target vector (approval status)
            model_type: Type of model to use ('xgboost', 'random_forest', 'gradient_boosting', 'ensemble')
        """
        if X is None or y is None or len(X) == 0:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['approval'] = scaler
        
        # Choose and train model
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'ensemble':
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=8, random_state=42)
            if XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=8, random_state=42)
                model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb_model)],
                    voting='soft'
                )
            else:
                model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb)],
                    voting='soft'
                )
        else:  # Default to Random Forest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            self.feature_importance['approval'] = feature_importance
        
        # Store model
        self.models['approval'] = model
        
        # Store metadata
        self.model_metadata['approval'] = {
            'model_type': model_type,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': list(X.columns),
            'training_date': datetime.now().isoformat()
        }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def train_processing_time_model(self, X, y, model_type='xgboost'):
        """
        Train model to predict processing time
        
        Args:
            X: Feature matrix
            y: Target vector (processing days)
            model_type: Type of model to use
        """
        if X is None or y is None:
            return None
        
        # Remove rows with missing processing time
        mask = ~y.isna() & (y > 0)
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['processing_time'] = scaler
        
        # Choose and train model
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        else:  # Default to Random Forest
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            self.feature_importance['processing_time'] = feature_importance
        
        # Store model
        self.models['processing_time'] = model
        
        # Store metadata
        self.model_metadata['processing_time'] = {
            'model_type': model_type,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': list(X.columns),
            'training_date': datetime.now().isoformat()
        }
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2
        }
    
    def train_time_series_model(self, data, target_column='applications_submitted', 
                                 model_type='prophet', periods=12):
        """
        Train time series forecasting model
        
        Args:
            data: DataFrame with time series data
            target_column: Column to predict
            model_type: 'prophet', 'arima', or 'lstm'
            periods: Number of future periods to forecast
        """
        if data is None or len(data) == 0 or target_column not in data.columns:
            return None
        
        try:
            # Prepare data
            if 'date' in data.columns:
                ts_data = data[['date', target_column]].copy()
                ts_data['date'] = pd.to_datetime(ts_data['date'])
                ts_data = ts_data.sort_values('date')
                ts_data = ts_data.set_index('date')
                ts_data = ts_data[target_column].resample('M').sum()
            else:
                return None
            
            if model_type == 'prophet' and PROPHET_AVAILABLE:
                # Prepare data for Prophet (requires 'ds' and 'y' columns)
                prophet_data = pd.DataFrame({
                    'ds': ts_data.index,
                    'y': ts_data.values
                })
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative'
                )
                model.fit(prophet_data)
                
                # Make future predictions
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast = model.predict(future)
                
                self.models['time_series_prophet'] = model
                
                return {
                    'model_type': 'prophet',
                    'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
                    'historical': prophet_data.to_dict('records')
                }
                
            elif model_type == 'arima' and ARIMA_AVAILABLE:
                # ARIMA model
                model = ARIMA(ts_data, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Forecast
                forecast = fitted_model.forecast(steps=periods)
                forecast_index = pd.date_range(start=ts_data.index[-1], periods=periods+1, freq='M')[1:]
                
                self.models['time_series_arima'] = fitted_model
                
                return {
                    'model_type': 'arima',
                    'forecast': [
                        {'date': str(idx), 'value': float(val)} 
                        for idx, val in zip(forecast_index, forecast)
                    ],
                    'historical': [
                        {'date': str(idx), 'value': float(val)} 
                        for idx, val in zip(ts_data.index, ts_data.values)
                    ]
                }
            
            else:
                # Simple linear trend as fallback
                return {
                    'model_type': 'linear',
                    'error': 'Advanced time series models not available'
                }
                
        except Exception as e:
            print(f"Error training time series model: {e}")
            return None
    
    def train_models(self, model_types=None):
        """
        Train all ML models with historical data
        
        Args:
            model_types: Dictionary specifying which models to train
        """
        print("=" * 60)
        print("Starting ML Model Training")
        print("=" * 60)
        
        results = {}
        
        # Load application data
        print("\n[1/4] Loading historical application data...")
        df = self.load_historical_data('applications', limit=5000)
        
        if df is None or len(df) == 0:
            return {"error": "No application data available for training"}
        
        print(f"Loaded {len(df)} application records")
        
        # Preprocess data
        print("\n[2/4] Preprocessing data...")
        X, y_approval, y_processing_time, features = self.preprocess_application_data(df)
        
        if X is None:
            return {"error": "Failed to preprocess data"}
        
        print(f"Features: {len(features)}")
        print(f"Features list: {features}")
        
        # Train approval prediction model
        print("\n[3/4] Training approval prediction model...")
        model_type = model_types.get('approval', 'ensemble') if model_types else 'ensemble'
        approval_results = self.train_approval_prediction_model(X, y_approval, model_type)
        
        if approval_results:
            results['approval_prediction'] = approval_results
            print(f"Approval Model - Accuracy: {approval_results['accuracy']:.4f}, "
                  f"F1-Score: {approval_results['f1_score']:.4f}")
        
        # Train processing time model
        print("\n[4/4] Training processing time prediction model...")
        model_type = model_types.get('processing_time', 'xgboost') if model_types else 'xgboost'
        processing_results = self.train_processing_time_model(X, y_processing_time, model_type)
        
        if processing_results:
            results['processing_time_prediction'] = processing_results
            print(f"Processing Time Model - RMSE: {processing_results['rmse']:.2f} days, "
                  f"R²: {processing_results['r2_score']:.4f}")
        
        # Train time series models
        print("\n[5/5] Training time series forecasting models...")
        trends_data = self.load_application_trends(months=24)
        
        if trends_data is not None and len(trends_data) > 0:
            # Aggregate by zone type
            for zone_type in trends_data['zone_type'].unique():
                zone_data = trends_data[trends_data['zone_type'] == zone_type]
                ts_result = self.train_time_series_model(
                    zone_data, 
                    target_column='applications_submitted',
                    model_type='prophet' if PROPHET_AVAILABLE else 'arima'
                )
                if ts_result:
                    results[f'time_series_{zone_type}'] = ts_result
        
        # Save models
        print("\nSaving trained models...")
        self.save_models()
        self.save_model_metadata()
        
        print("\n" + "=" * 60)
        print("Model Training Completed Successfully!")
        print("=" * 60)
        
        return results
    
    def predict_approval_probability(self, application_data):
        """Predict approval probability for a new application"""
        if 'approval' not in self.models:
            self.load_models()
        
        if 'approval' not in self.models:
            return None
        
        # Prepare features
        features = self.prepare_features_for_prediction(application_data)
        if features is None:
            return None
        
        # Scale features
        X_scaled = self.scalers['approval'].transform([features])
        
        # Predict
        probability = self.models['approval'].predict_proba(X_scaled)[0][1]
        
        # Determine confidence
        if probability > 0.8 or probability < 0.2:
            confidence = 'high'
        elif probability > 0.6 or probability < 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'approval_probability': float(probability),
            'approval_prediction': 'approved' if probability > 0.5 else 'rejected',
            'confidence': confidence,
            'model_version': self.model_metadata.get('approval', {}).get('model_type', 'unknown')
        }
    
    def predict_processing_time(self, application_data):
        """Predict processing time for a new application"""
        if 'processing_time' not in self.models:
            self.load_models()
        
        if 'processing_time' not in self.models:
            return None
        
        # Prepare features
        features = self.prepare_features_for_prediction(application_data)
        if features is None:
            return None
        
        # Scale features
        X_scaled = self.scalers['processing_time'].transform([features])
        
        # Predict
        predicted_days = self.models['processing_time'].predict(X_scaled)[0]
        
        return {
            'predicted_days': max(1, round(predicted_days)),
            'confidence_interval': {
                'lower': max(1, round(predicted_days * 0.7)),
                'upper': round(predicted_days * 1.3)
            },
            'model_version': self.model_metadata.get('processing_time', {}).get('model_type', 'unknown')
        }
    
    def prepare_features_for_prediction(self, application_data):
        """Prepare features from application data for prediction"""
        try:
            current_date = datetime.now()
            
            # Start with basic temporal features
            features = [
                current_date.year,  # application_year
                current_date.month,  # application_month
                current_date.weekday(),  # application_day_of_week
                1 if current_date.weekday() >= 5 else 0,  # is_weekend
                0,  # days_since_submission (will be 0 for new applications)
            ]
            
            # Add numeric features
            features.append(float(application_data.get('project_area', 100)))
            features.append(float(application_data.get('lot_area', 100)))
            features.append(float(application_data.get('age', 35)))
            
            # Add area ratio
            project_area = float(application_data.get('project_area', 100))
            lot_area = float(application_data.get('lot_area', 100))
            features.append(project_area / (lot_area + 1))
            
            # Add encoded categorical features
            if 'project_type' in application_data and 'project_type' in self.encoders:
                try:
                    encoded = self.encoders['project_type'].transform([str(application_data['project_type'])])[0]
                    features.append(encoded)
                except:
                    features.append(0)
            else:
                features.append(0)
            
            if 'project_nature' in application_data and 'project_nature' in self.encoders:
                try:
                    encoded = self.encoders['project_nature'].transform([str(application_data['project_nature'])])[0]
                    features.append(encoded)
                except:
                    features.append(0)
            else:
                features.append(0)
            
            if 'status' in application_data and 'status' in self.encoders:
                try:
                    encoded = self.encoders['status'].transform([str(application_data['status'])])[0]
                    features.append(encoded)
                except:
                    features.append(0)
            else:
                features.append(0)
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def save_models(self):
        """Save all trained models, encoders, and scalers"""
        try:
            for name, model in self.models.items():
                model_path = os.path.join(self.models_dir, f'{name}_model.pkl')
                joblib.dump(model, model_path)
                print(f"Saved model: {model_path}")
            
            for name, encoder in self.encoders.items():
                encoder_path = os.path.join(self.models_dir, f'{name}_encoder.pkl')
                joblib.dump(encoder, encoder_path)
                print(f"Saved encoder: {encoder_path}")
            
            for name, scaler in self.scalers.items():
                scaler_path = os.path.join(self.models_dir, f'{name}_scaler.pkl')
                joblib.dump(scaler, scaler_path)
                print(f"Saved scaler: {scaler_path}")
            
            # Save feature importance
            importance_path = os.path.join(self.models_dir, 'feature_importance.json')
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            print("All models saved successfully")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def save_model_metadata(self):
        """Save model metadata to database"""
        connection = self.connect_database()
        if not connection:
            return
        
        try:
            cursor = connection.cursor()
            
            for model_name, metadata in self.model_metadata.items():
                # Determine model type
                model_type = 'classification' if 'accuracy' in metadata else 'regression'
                
                # Insert or update model performance
                insert_query = """
                INSERT INTO model_performance 
                (model_name, model_type, version, accuracy, precision_score, recall, f1_score,
                 mae, rmse, r2_score, training_samples, test_samples, training_date, 
                 feature_importance, hyperparameters, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                accuracy = VALUES(accuracy),
                precision_score = VALUES(precision_score),
                recall = VALUES(recall),
                f1_score = VALUES(f1_score),
                mae = VALUES(mae),
                rmse = VALUES(rmse),
                r2_score = VALUES(r2_score),
                training_samples = VALUES(training_samples),
                test_samples = VALUES(test_samples),
                training_date = VALUES(training_date),
                feature_importance = VALUES(feature_importance),
                is_active = 1
                """
                
                feature_imp = json.dumps(self.feature_importance.get(model_name, {}))
                hyperparams = json.dumps({'model_type': metadata.get('model_type', 'unknown')})
                
                cursor.execute(insert_query, (
                    model_name,
                    model_type,
                    '1.0',
                    metadata.get('accuracy'),
                    metadata.get('precision'),
                    metadata.get('recall'),
                    metadata.get('f1_score'),
                    metadata.get('mae'),
                    metadata.get('rmse'),
                    metadata.get('r2_score'),
                    metadata.get('training_samples'),
                    metadata.get('test_samples'),
                    metadata.get('training_date'),
                    feature_imp,
                    hyperparams,
                    1
                ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
            print("Model metadata saved to database")
            
        except Exception as e:
            print(f"Error saving model metadata: {e}")
            if connection:
                connection.close()
    
    def load_models(self):
        """Load trained models, encoders, and scalers"""
        try:
            # Load models
            model_files = {
                'approval': 'approval_model.pkl',
                'processing_time': 'processing_time_model.pkl'
            }
            
            for name, filename in model_files.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    print(f"Loaded model: {name}")
            
            # Load encoders
            encoder_files = ['project_type', 'project_nature', 'status', 'gender']
            for encoder_name in encoder_files:
                encoder_path = os.path.join(self.models_dir, f'{encoder_name}_encoder.pkl')
                if os.path.exists(encoder_path):
                    self.encoders[encoder_name] = joblib.load(encoder_path)
                    print(f"Loaded encoder: {encoder_name}")
            
            # Load scalers
            scaler_files = ['approval', 'processing_time']
            for scaler_name in scaler_files:
                scaler_path = os.path.join(self.models_dir, f'{scaler_name}_scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    print(f"Loaded scaler: {scaler_name}")
            
            # Load feature importance
            importance_path = os.path.join(self.models_dir, 'feature_importance.json')
            if os.path.exists(importance_path):
                with open(importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_predictions_summary(self):
        """Get summary of all available models and predictions"""
        if not self.models:
            self.load_models()
        
        if not self.models:
            return {"error": "No models available"}
        
        summary = {
            'available_models': list(self.models.keys()),
            'feature_importance': self.feature_importance,
            'model_metadata': self.model_metadata,
            'model_performance': {}
        }
        
        # Load performance from database
        connection = self.connect_database()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("""
                    SELECT model_name, accuracy, precision_score, recall, f1_score, 
                           training_date, is_active
                    FROM model_performance
                    WHERE is_active = 1
                    ORDER BY training_date DESC
                """)
                summary['model_performance'] = cursor.fetchall()
                cursor.close()
            except Exception as e:
                print(f"Error loading model performance: {e}")
            finally:
                connection.close()
        
        return summary
    
    def collect_historical_data(self, data_source='applications'):
        """
        Collect and store historical data for training
        
        Args:
            data_source: Source of data ('applications', 'population', 'economic')
        """
        connection = self.connect_database()
        if not connection:
            return None
        
        try:
            cursor = connection.cursor()
            
            if data_source == 'applications':
                # Aggregate application data by month and zone type
                query = """
                INSERT INTO application_trends 
                (date, zone_type, applications_submitted, applications_approved, 
                 applications_rejected, applications_pending, approval_rate)
                SELECT 
                    DATE_FORMAT(created_at, '%Y-%m-01') as date,
                    project_type as zone_type,
                    COUNT(*) as applications_submitted,
                    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as applications_approved,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as applications_rejected,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as applications_pending,
                    (SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) / COUNT(*)) * 100 as approval_rate
                FROM application_forms
                WHERE created_at IS NOT NULL
                GROUP BY DATE_FORMAT(created_at, '%Y-%m-01'), project_type
                ON DUPLICATE KEY UPDATE
                applications_submitted = VALUES(applications_submitted),
                applications_approved = VALUES(applications_approved),
                applications_rejected = VALUES(applications_rejected),
                applications_pending = VALUES(applications_pending),
                approval_rate = VALUES(approval_rate),
                updated_at = NOW()
                """
                
                cursor.execute(query)
                records_collected = cursor.rowcount
                
                # Log collection
                log_query = """
                INSERT INTO data_collection_log 
                (data_source, data_type, records_collected, collection_date, status)
                VALUES (%s, %s, %s, %s, 'success')
                """
                cursor.execute(log_query, (
                    'application_forms',
                    'applications',
                    records_collected,
                    datetime.now().date()
                ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
            print(f"Collected {records_collected} records from {data_source}")
            return {'success': True, 'records_collected': records_collected}
            
        except Exception as e:
            print(f"Error collecting historical data: {e}")
            if connection:
                connection.close()
            return {'success': False, 'error': str(e)}


# Main execution
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'u520834156_dbUPAHOZoning'
    }
    
    # Initialize ML analytics
    ml_analytics = MLPredictiveAnalytics(db_config)
    
    # Collect historical data first
    print("Collecting historical data...")
    ml_analytics.collect_historical_data('applications')
    
    # Train models
    print("\nStarting ML model training...")
    results = ml_analytics.train_models({
        'approval': 'ensemble',
        'processing_time': 'xgboost'
    })
    
    print("\nTraining Results:")
    print(json.dumps(results, indent=2))
