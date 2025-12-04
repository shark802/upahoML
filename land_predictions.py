#!/usr/bin/env python3
"""
Land Use and Land Cost Prediction Models using Linear Regression
UPAHO Zoning Management System
"""

import pandas as pd
import numpy as np
import joblib
import mysql.connector
from datetime import datetime, timedelta
import json
import os
import warnings
import math
warnings.filterwarnings('ignore')

# Linear Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Time Series Forecasting - ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Statsmodels not available. ARIMA forecasting will not work.")


class LandPredictions:
    """
    Linear Regression Models for Land Use and Land Cost Predictions
    """
    
    def __init__(self, db_config):
        """Initialize Land Predictions with database configuration"""
        self.db_config = db_config
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        # Use models directory within app (not ../models which is read-only on Heroku)
        self.models_dir = os.path.join(self.base_path, 'models')
        
        # Try to create models directory, but don't fail if it's read-only
        try:
            os.makedirs(self.models_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            # On Heroku, if we can't create, try /tmp/models as fallback
            if 'read-only' in str(e).lower() or 'errno 30' in str(e):
                self.models_dir = os.path.join('/tmp', 'models')
                os.makedirs(self.models_dir, exist_ok=True)
            else:
                raise
        
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
    
    def load_land_data(self):
        """Load land use and cost data from database with enhanced features"""
        connection = self.connect_database()
        if not connection:
            return None
            
        try:
            query = """
            SELECT 
                af.id,
                af.project_type,
                af.project_nature,
                af.project_location,
                af.project_area,
                af.lot_area,
                af.project_cost_numeric,
                af.created_at,
                af.latitude,
                af.longitude,
                af.site_zoning,
                af.location_type,
                af.land_uses,
                YEAR(af.created_at) as year,
                MONTH(af.created_at) as month,
                c.age,
                c.gender
            FROM application_forms af
            LEFT JOIN clients c ON af.client_id = c.id
            WHERE af.project_type IS NOT NULL
            AND af.project_cost_numeric IS NOT NULL
            AND af.created_at IS NOT NULL
            AND af.project_area IS NOT NULL
            AND af.lot_area IS NOT NULL
            ORDER BY af.created_at DESC
            LIMIT 5000
            """
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            if len(df) == 0:
                return None
            
            # Calculate cost per square meter
            df['cost_per_sqm'] = None
            mask = (df['project_cost_numeric'].notna()) & (df['lot_area'].notna()) & (df['lot_area'] > 0)
            df.loc[mask, 'cost_per_sqm'] = df.loc[mask, 'project_cost_numeric'] / df.loc[mask, 'lot_area']
            
            return df
            
        except Exception as e:
            print(f"Error loading land data: {e}")
            if connection:
                connection.close()
            return None
    
    def _engineer_features(self, df):
        """Enhanced feature engineering for land property value prediction"""
        df = df.copy()
        
        # 1. SIZE FEATURES
        # Area ratio (building density)
        df['area_ratio'] = df['project_area'] / (df['lot_area'] + 1e-6)
        df['area_ratio'] = df['area_ratio'].clip(0, 1)
        
        # Size difference
        df['size_difference'] = df['lot_area'] - df['project_area']
        
        # Efficiency ratio
        df['efficiency_ratio'] = df['project_area'] / (df['lot_area'] + 1e-6)
        
        # Lot size categories (encoded as numeric)
        df['lot_size_small'] = (df['lot_area'] < 100).astype(int)
        df['lot_size_medium'] = ((df['lot_area'] >= 100) & (df['lot_area'] < 500)).astype(int)
        df['lot_size_large'] = (df['lot_area'] >= 500).astype(int)
        
        # 2. LOCATION FEATURES
        # Distance to city center (if coordinates available)
        city_center_lat = 14.5995  # Adjust to your city center
        city_center_lon = 120.9842
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            def calc_distance(row):
                if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                    try:
                        lat1, lon1 = float(row['latitude']), float(row['longitude'])
                        lat2, lon2 = city_center_lat, city_center_lon
                        R = 6371  # Earth radius in km
                        dlat = math.radians(lat2 - lat1)
                        dlon = math.radians(lon2 - lon1)
                        a = (math.sin(dlat/2) * math.sin(dlat/2) +
                             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                             math.sin(dlon/2) * math.sin(dlon/2))
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        return R * c
                    except:
                        return None
                return None
            
            df['distance_to_center'] = df.apply(calc_distance, axis=1)
            df['distance_to_center'].fillna(df['distance_to_center'].median(), inplace=True)
        
        # Location category encoding
        location_mapping = {
            'Downtown': 3, 'Urban Core': 3, 'Commercial District': 3,
            'Suburban': 2, 'Residential Area': 2, 'Mixed Zone': 2,
            'Industrial Zone': 1, 'Rural': 1, 'Agricultural': 1
        }
        if 'project_location' in df.columns:
            df['location_category'] = df['project_location'].map(location_mapping).fillna(2)
        
        # 3. TEMPORAL FEATURES
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['year'] = df['created_at'].dt.year
            df['month'] = df['created_at'].dt.month
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['quarter'] = df['created_at'].dt.quarter
            
            # Cyclical encoding for month (captures seasonality)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 4. ZONING/TYPE FEATURES
        # Project type encoding (keep existing)
        if 'project_type' in df.columns:
            if 'project_type' not in self.encoders:
                self.encoders['project_type'] = LabelEncoder()
                df['project_type_encoded'] = self.encoders['project_type'].fit_transform(
                    df['project_type'].astype(str).fillna('residential')
                )
            else:
                try:
                    df['project_type_encoded'] = self.encoders['project_type'].transform(
                        df['project_type'].astype(str).fillna('residential')
                    )
                except:
                    known = set(self.encoders['project_type'].classes_)
                    unknown = set(df['project_type'].astype(str).unique()) - known
                    for cat in unknown:
                        self.encoders['project_type'].classes_ = np.append(
                            self.encoders['project_type'].classes_, cat
                        )
                    df['project_type_encoded'] = self.encoders['project_type'].transform(
                        df['project_type'].astype(str).fillna('residential')
                    )
        
        # Zoning category (if available)
        if 'site_zoning' in df.columns:
            zoning_mapping = {
                'Commercial': 3, 'Mixed-Use': 3, 'Residential-High': 3,
                'Residential': 2, 'Institutional': 2,
                'Agricultural': 1, 'Industrial': 1, 'Rural': 1
            }
            df['zoning_category'] = df['site_zoning'].map(zoning_mapping).fillna(2)
        
        # 5. INTERACTION FEATURES
        # Size × Location interaction
        if 'lot_area' in df.columns and 'location_category' in df.columns:
            df['size_location_interaction'] = df['lot_area'] * df['location_category']
        
        # Type × Size interaction
        if 'project_type_encoded' in df.columns and 'lot_area' in df.columns:
            df['type_size_interaction'] = df['project_type_encoded'] * df['lot_area']
        
        # Year × Location interaction
        if 'year' in df.columns and 'location_category' in df.columns:
            base_year = df['year'].min()
            df['year_location_interaction'] = (df['year'] - base_year) * df['location_category']
        
        return df
    
    def preprocess_land_cost_data(self, df):
        """Preprocess data for land cost prediction with enhanced features"""
        if df is None or len(df) == 0:
            return None, None, None
        
        df = df.copy()
        
        # Remove rows with missing cost data
        df = df[df['cost_per_sqm'].notna() & (df['cost_per_sqm'] > 0)]
        
        if len(df) == 0:
            return None, None, None
        
        # Handle missing values for numeric columns
        numeric_cols = ['lot_area', 'project_area', 'age', 'year', 'month']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in ['lot_area', 'project_area']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Apply feature engineering
        df = self._engineer_features(df)
        
        # Select all engineered features
        feature_columns = [
            # Basic features
            'lot_area', 'project_area', 'year', 'month', 'age',
            
            # Size features
            'area_ratio', 'size_difference', 'efficiency_ratio',
            'lot_size_small', 'lot_size_medium', 'lot_size_large',
            
            # Location features
            'location_category',
            
            # Temporal features
            'day_of_week', 'is_weekend', 'quarter',
            'month_sin', 'month_cos',
            
            # Type features
            'project_type_encoded',
            
            # Zoning features
            'zoning_category',
            
            # Interaction features
            'size_location_interaction', 'type_size_interaction',
            'year_location_interaction'
        ]
        
        # Add distance feature if available
        if 'distance_to_center' in df.columns:
            feature_columns.append('distance_to_center')
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].fillna(0)
        
        # Target
        y = df['cost_per_sqm']
        
        # Remove outliers (cost per sqm)
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (y >= lower_bound) & (y <= upper_bound)
        
        X = X[mask]
        y = y[mask]
        
        return X, y, available_features
    
    def train_land_cost_model(self, model_type='random_forest'):
        """
        Train model to predict land cost per square meter
        
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'polynomial', 
                       'random_forest', 'gradient_boosting', 'ensemble'
        """
        print("Loading land data for cost prediction...")
        df = self.load_land_data()
        
        if df is None or len(df) == 0:
            return {"error": "No land cost data available"}
        
        print(f"Loaded {len(df)} records")
        
        # Preprocess data
        X, y, features = self.preprocess_land_cost_data(df)
        
        if X is None or len(X) == 0:
            return {"error": "No valid data after preprocessing"}
        
        print(f"Training samples: {len(X)}")
        print(f"Features: {features}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['land_cost'] = scaler
        
        # Choose and train model
        if model_type == 'random_forest':
            # Random Forest - Best balance of accuracy and realism
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            # Gradient Boosting - High accuracy
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif model_type == 'ensemble':
            # Ensemble - Most realistic (combines multiple models)
            rf = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            gb = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            model = VotingRegressor(
                estimators=[('rf', rf), ('gb', gb)],
                weights=[0.5, 0.5]
            )
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
        elif model_type == 'polynomial':
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_scaled = poly.fit_transform(X_train_scaled)
            X_test_scaled = poly.transform(X_test_scaled)
            model = LinearRegression()
            self.models['land_cost_poly'] = poly
        else:  # Linear regression (default fallback)
            model = LinearRegression()
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store model
        if model_type == 'polynomial':
            self.models['land_cost'] = model
        else:
            self.models['land_cost'] = model
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            # For tree-based models (Random Forest, Gradient Boosting)
            feature_importance = dict(zip(features, model.feature_importances_))
            self.feature_importance['land_cost'] = feature_importance
        elif hasattr(model, 'coef_'):
            # For linear models (coefficients)
            feature_importance = dict(zip(features, abs(model.coef_)))
            self.feature_importance['land_cost'] = feature_importance
        elif hasattr(model, 'estimators_'):
            # For ensemble models (average feature importance)
            importances = []
            for estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
                feature_importance = dict(zip(features, avg_importance))
                self.feature_importance['land_cost'] = feature_importance
        
        # Store metadata
        self.model_metadata = {
            'land_cost': {
                'model_type': model_type,
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': features,
                'training_date': datetime.now().isoformat()
            }
        }
        
        print(f"\nLand Cost Model Training Complete:")
        print(f"  RMSE: {rmse:.2f} PHP/sqm")
        print(f"  MAE: {mae:.2f} PHP/sqm")
        print(f"  R² Score: {r2:.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'features': features
        }
    
    def predict_land_cost(self, land_data):
        """
        Predict land cost per square meter with enhanced features
        
        Args:
            land_data: Dict with land property details including:
                - Basic: 'lot_area', 'project_area', 'project_type', 'year', 'month', 'age'
                - Location: 'location', 'latitude', 'longitude'
                - Zoning: 'site_zoning', 'location_type'
        """
        if 'land_cost' not in self.models:
            self.load_models()
        
        if 'land_cost' not in self.models:
            return None
        
        try:
            # Create a DataFrame row for feature engineering
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            # Build feature dict
            feature_dict = {
                'lot_area': float(land_data.get('lot_area', 100)),
                'project_area': float(land_data.get('project_area', 100)),
                'year': int(land_data.get('year', current_year)),
                'month': int(land_data.get('month', current_month)),
                'age': float(land_data.get('age', 35)),
                'project_type': str(land_data.get('project_type', 'residential')),
                'project_location': str(land_data.get('location', land_data.get('project_location', ''))),
                'created_at': pd.Timestamp(year=land_data.get('year', current_year), 
                                         month=land_data.get('month', current_month), day=1)
            }
            
            # Add optional features if available
            if 'latitude' in land_data:
                feature_dict['latitude'] = float(land_data.get('latitude', 0))
            if 'longitude' in land_data:
                feature_dict['longitude'] = float(land_data.get('longitude', 0))
            if 'site_zoning' in land_data:
                feature_dict['site_zoning'] = str(land_data.get('site_zoning', ''))
            if 'location_type' in land_data:
                feature_dict['location_type'] = str(land_data.get('location_type', ''))
            
            # Create DataFrame with single row
            df = pd.DataFrame([feature_dict])
            
            # Apply feature engineering
            df = self._engineer_features(df)
            
            # Get feature list from model metadata (use same features as training)
            if 'land_cost' in self.model_metadata:
                expected_features = self.model_metadata['land_cost'].get('features', [])
            else:
                # Fallback to default features
                expected_features = [
                    'lot_area', 'project_area', 'year', 'month', 'age',
                    'area_ratio', 'size_difference', 'efficiency_ratio',
                    'lot_size_small', 'lot_size_medium', 'lot_size_large',
                    'location_category', 'day_of_week', 'is_weekend', 'quarter',
                    'month_sin', 'month_cos', 'project_type_encoded',
                    'size_location_interaction', 'type_size_interaction',
                    'year_location_interaction'
                ]
            
            # Prepare features in correct order
            features = []
            for feat in expected_features:
                if feat in df.columns:
                    val = df[feat].iloc[0]
                    if pd.isna(val):
                        features.append(0)
                    else:
                        features.append(float(val))
                else:
                    features.append(0)  # Missing feature, use default
            
            # Scale features
            X_scaled = self.scalers['land_cost'].transform([features])
            
            # Apply polynomial if needed
            if 'land_cost_poly' in self.models:
                X_scaled = self.models['land_cost_poly'].transform(X_scaled)
            
            # Predict
            predicted_cost = self.models['land_cost'].predict(X_scaled)[0]
            
            return {
                'predicted_cost_per_sqm': max(0, float(predicted_cost)),
                'confidence': 'high' if self.model_metadata.get('land_cost', {}).get('r2_score', 0) > 0.7 else 'medium',
                'model_r2': self.model_metadata.get('land_cost', {}).get('r2_score', 0),
                'features_used': len(expected_features)
            }
            
        except Exception as e:
            print(f"Error predicting land cost: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_land_use_trends(self):
        """Load historical land use trends for forecasting"""
        connection = self.connect_database()
        if not connection:
            return None
        
        try:
            query = """
            SELECT 
                DATE_FORMAT(created_at, '%Y-%m') as month,
                project_type,
                COUNT(*) as count,
                AVG(lot_area) as avg_area,
                AVG(project_cost_numeric / NULLIF(lot_area, 0)) as avg_cost_per_sqm
            FROM application_forms
            WHERE project_type IS NOT NULL
            AND created_at IS NOT NULL
            AND created_at >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
            GROUP BY DATE_FORMAT(created_at, '%Y-%m'), project_type
            ORDER BY month, project_type
            """
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            return df
            
        except Exception as e:
            print(f"Error loading land use trends: {e}")
            if connection:
                connection.close()
            return None
    
    def predict_future_land_use(self, months_ahead=12):
        """
        Predict future land use distribution using linear regression
        
        Args:
            months_ahead: Number of months to forecast
        """
        df = self.load_land_use_trends()
        
        if df is None or len(df) == 0:
            return None
        
        # Convert month to numeric for regression
        df['month_num'] = pd.to_datetime(df['month']).astype('int64') // 10**9
        
        # Group by project_type and fit linear regression for each
        predictions = {}
        land_use_types = df['project_type'].unique()
        
        for land_use in land_use_types:
            land_use_data = df[df['project_type'] == land_use].sort_values('month')
            
            if len(land_use_data) < 3:
                continue
            
            X = land_use_data[['month_num']].values
            y = land_use_data['count'].values
            
            # Train linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future months
            last_month = pd.to_datetime(land_use_data['month'].max())
            future_months = []
            future_counts = []
            
            for i in range(1, months_ahead + 1):
                future_date = last_month + pd.DateOffset(months=i)
                future_month_num = pd.to_datetime(future_date).value // 10**9
                
                predicted_count = model.predict([[future_month_num]])[0]
                predicted_count = max(0, int(predicted_count))
                
                future_months.append(future_date.strftime('%Y-%m'))
                future_counts.append(predicted_count)
            
            predictions[land_use] = {
                'historical': [
                    {'month': row['month'], 'count': int(row['count'])} 
                    for _, row in land_use_data.iterrows()
                ],
                'forecast': [
                    {'month': month, 'predicted_count': count}
                    for month, count in zip(future_months, future_counts)
                ],
                'trend': 'increasing' if model.coef_[0] > 0 else 'decreasing',
                'growth_rate': float(model.coef_[0])
            }
        
        return predictions
    
    def save_models(self):
        """Save trained models"""
        try:
            for name, model in self.models.items():
                if name != 'land_cost_poly':  # Save poly separately if needed
                    model_path = os.path.join(self.models_dir, f'{name}_model.pkl')
                    joblib.dump(model, model_path)
                    print(f"Saved model: {model_path}")
            
            # Save polynomial transformer if exists
            if 'land_cost_poly' in self.models:
                poly_path = os.path.join(self.models_dir, 'land_cost_poly.pkl')
                joblib.dump(self.models['land_cost_poly'], poly_path)
                print(f"Saved polynomial transformer: {poly_path}")
            
            # Save encoders
            for name, encoder in self.encoders.items():
                encoder_path = os.path.join(self.models_dir, f'{name}_encoder.pkl')
                joblib.dump(encoder, encoder_path)
                print(f"Saved encoder: {encoder_path}")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                scaler_path = os.path.join(self.models_dir, f'{name}_scaler.pkl')
                joblib.dump(scaler, scaler_path)
                print(f"Saved scaler: {scaler_path}")
            
            # Save feature importance
            importance_path = os.path.join(self.models_dir, 'land_feature_importance.json')
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, verbose=False):
        """Load trained models"""
        try:
            # Load land cost model
            model_path = os.path.join(self.models_dir, 'land_cost_model.pkl')
            if os.path.exists(model_path):
                self.models['land_cost'] = joblib.load(model_path)
                if verbose:
                    print("Loaded land_cost model")
            
            # Load polynomial transformer if exists
            poly_path = os.path.join(self.models_dir, 'land_cost_poly.pkl')
            if os.path.exists(poly_path):
                self.models['land_cost_poly'] = joblib.load(poly_path)
                if verbose:
                    print("Loaded polynomial transformer")
            
            # Load encoders
            encoder_path = os.path.join(self.models_dir, 'project_type_encoder.pkl')
            if os.path.exists(encoder_path):
                self.encoders['project_type'] = joblib.load(encoder_path)
                if verbose:
                    print("Loaded project_type encoder")
            
            # Load scalers
            scaler_path = os.path.join(self.models_dir, 'land_cost_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scalers['land_cost'] = joblib.load(scaler_path)
                if verbose:
                    print("Loaded land_cost scaler")
            
            # Load feature importance
            importance_path = os.path.join(self.models_dir, 'land_feature_importance.json')
            if os.path.exists(importance_path):
                with open(importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
            
            return len(self.models) > 0
            
        except Exception as e:
            if verbose:
                print(f"Error loading models: {e}")
            return False
    
    def save_model_metadata(self):
        """Save model metadata to database"""
        connection = self.connect_database()
        if not connection:
            return
        
        try:
            cursor = connection.cursor()
            
            if 'land_cost' in self.model_metadata:
                metadata = self.model_metadata['land_cost']
                
                insert_query = """
                INSERT INTO model_performance 
                (model_name, model_type, version, mae, rmse, r2_score, training_samples, 
                 test_samples, training_date, feature_importance, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                mae = VALUES(mae),
                rmse = VALUES(rmse),
                r2_score = VALUES(r2_score),
                training_samples = VALUES(training_samples),
                test_samples = VALUES(test_samples),
                training_date = VALUES(training_date),
                feature_importance = VALUES(feature_importance),
                is_active = 1
                """
                
                feature_imp = json.dumps(self.feature_importance.get('land_cost', {}))
                
                cursor.execute(insert_query, (
                    'land_cost',
                    'regression',
                    '1.0',
                    metadata.get('mae'),
                    metadata.get('rmse'),
                    metadata.get('r2_score'),
                    metadata.get('training_samples'),
                    metadata.get('test_samples'),
                    metadata.get('training_date'),
                    feature_imp,
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
    
    def calculate_location_factors(self):
        """Calculate location premium/discount factors from historical data"""
        connection = self.connect_database()
        if not connection:
            return {}
        
        try:
            query = """
                SELECT 
                    project_location,
                    AVG(project_cost_numeric / NULLIF(lot_area, 0)) as avg_cost_per_sqm,
                    COUNT(*) as count
                FROM application_forms
                WHERE project_location IS NOT NULL
                AND project_location != ''
                AND project_cost_numeric IS NOT NULL
                AND lot_area IS NOT NULL
                AND lot_area > 0
                GROUP BY project_location
                HAVING count >= 5
            """
            
            df = pd.read_sql(query, connection)
            connection.close()
            
            if len(df) == 0:
                return {}
            
            # Calculate overall average
            overall_avg = df['avg_cost_per_sqm'].mean()
            
            # Calculate location factors
            location_factors = {}
            for _, row in df.iterrows():
                factor = row['avg_cost_per_sqm'] / overall_avg if overall_avg > 0 else 1.0
                location_factors[row['project_location']] = {
                    'factor': float(factor),
                    'avg_cost_per_sqm': float(row['avg_cost_per_sqm']),
                    'count': int(row['count'])
                }
            
            return location_factors
            
        except Exception as e:
            print(f"Error calculating location factors: {e}")
            if connection:
                connection.close()
            return {}
    
    def calculate_yearly_appreciation_rate(self, project_type=None, location=None):
        """
        Calculate realistic yearly appreciation/depreciation rate from historical data
        Returns positive value for increase, negative for decrease
        """
        connection = self.connect_database()
        if not connection:
            return 0.03  # Default 3% if no data
        
        try:
            query = """
                SELECT 
                    YEAR(created_at) as year,
                    AVG(project_cost_numeric / NULLIF(lot_area, 0)) as avg_cost_per_sqm,
                    COUNT(*) as count
                FROM application_forms
                WHERE project_cost_numeric IS NOT NULL
                AND lot_area IS NOT NULL
                AND lot_area > 0
                AND created_at IS NOT NULL
                AND created_at >= DATE_SUB(CURDATE(), INTERVAL 5 YEAR)
            """
            
            params = []
            if project_type:
                query += " AND project_type = %s"
                params.append(project_type)
            
            if location:
                query += " AND project_location = %s"
                params.append(location)
            
            query += " GROUP BY YEAR(created_at) HAVING count >= 3 ORDER BY year"
            
            df = pd.read_sql(query, connection, params=params)
            connection.close()
            
            if len(df) < 2:
                # Not enough data, use default based on location/project type
                if location and any(word in location.lower() for word in ['rural', 'agricultural']):
                    return -0.01  # Slight decrease for rural areas
                return 0.03  # Default 3% increase
            
            # Calculate year-over-year growth
            df = df.sort_values('year')
            df['growth'] = df['avg_cost_per_sqm'].pct_change()
            
            # Calculate statistics
            growth_rates = df['growth'].dropna()
            
            if len(growth_rates) == 0:
                return 0.03
            
            # Use median for more robust estimate (less affected by outliers)
            median_growth = growth_rates.median()
            mean_growth = growth_rates.mean()
            
            # Use weighted average (more recent years have more weight)
            if len(growth_rates) > 1:
                weights = np.linspace(0.5, 1.0, len(growth_rates))
                weighted_avg = np.average(growth_rates, weights=weights)
            else:
                weighted_avg = mean_growth
            
            # Prefer median if there's high volatility, otherwise use weighted average
            std_dev = growth_rates.std()
            if std_dev > 0.1:  # High volatility
                final_rate = median_growth
            else:
                final_rate = weighted_avg
            
            # Cap extreme values (between -10% and +15% per year)
            final_rate = max(-0.10, min(0.15, final_rate))
            
            # Return as decimal (positive = increase, negative = decrease)
            return float(final_rate) if not pd.isna(final_rate) else 0.03
            
        except Exception as e:
            print(f"Error calculating appreciation rate: {e}")
            if connection:
                connection.close()
            return 0.03
    
    def predict_land_cost_future(self, land_data, target_years=10):
        """
        Predict land cost for future years (5-10 years) considering location
        
        Args:
            land_data: Dict with 'lot_area', 'project_area', 'project_type', 
                      'location', 'year', 'month', 'age'
            target_years: Number of years in the future to predict (default 10)
        
        Returns:
            Dict with current and future predictions, yearly breakdown
        """
        from datetime import datetime
        
        # Get current year prediction
        current_year = datetime.now().year
        land_data_current = land_data.copy()
        land_data_current['year'] = current_year
        
        current_prediction = self.predict_land_cost(land_data_current)
        if not current_prediction:
            return None
        
        current_cost = current_prediction['predicted_cost_per_sqm']
        
        # Get location factor
        location = land_data.get('location', '')
        location_factors = self.calculate_location_factors()
        location_factor = 1.0
        location_category = 'standard'
        
        if location and location in location_factors:
            location_factor = location_factors[location]['factor']
            # Categorize location
            if location_factor >= 1.2:
                location_category = 'premium'
            elif location_factor <= 0.85:
                location_category = 'economy'
            else:
                location_category = 'standard'
        else:
            # Try to match location patterns
            location_lower = location.lower() if location else ''
            if any(word in location_lower for word in ['downtown', 'urban core', 'commercial district']):
                location_factor = 1.25
                location_category = 'premium'
            elif any(word in location_lower for word in ['rural', 'agricultural']):
                location_factor = 0.80
                location_category = 'economy'
        
        # Apply location factor to current prediction
        adjusted_current_cost = current_cost * location_factor
        
        # Get appreciation rate (can be positive or negative)
        project_type = land_data.get('project_type', 'residential')
        base_appreciation_rate = self.calculate_yearly_appreciation_rate(project_type, location)
        
        # Adjust appreciation rate based on location category (but allow negative)
        if location_category == 'premium':
            appreciation_rate = base_appreciation_rate * 1.15  # Premium locations appreciate faster
        elif location_category == 'economy':
            appreciation_rate = base_appreciation_rate * 0.85  # Economy locations appreciate slower
        else:
            appreciation_rate = base_appreciation_rate
        
        # Determine trend direction
        is_increasing = appreciation_rate > 0
        trend_direction = 'increasing' if is_increasing else 'decreasing'
        
        # Calculate future predictions (handles both increase and decrease)
        yearly_breakdown = []
        for year_offset in range(target_years + 1):
            year = current_year + year_offset
            # Compound growth: cost * (1 + rate)^years (works for negative rates too)
            future_cost = adjusted_current_cost * ((1 + appreciation_rate) ** year_offset)
            # Ensure cost doesn't go below 10% of original (minimum floor)
            future_cost = max(adjusted_current_cost * 0.1, future_cost)
            
            yearly_breakdown.append({
                'year': int(year),
                'cost_per_sqm': float(future_cost),
                'total_value': float(future_cost * land_data.get('lot_area', 100)),
                'change_from_current': float((future_cost / adjusted_current_cost) - 1)
            })
        
        target_cost = yearly_breakdown[-1]['cost_per_sqm']
        total_appreciation = (target_cost / adjusted_current_cost) - 1
        
        # Create realistic scenarios based on actual rate
        if appreciation_rate > 0:
            # Positive trend - scenarios around the rate
            scenarios = {
                'optimistic': {
                    'rate': appreciation_rate * 1.5,  # 50% higher growth
                    'cost_per_sqm': float(adjusted_current_cost * ((1 + appreciation_rate * 1.5) ** target_years)),
                    'total_appreciation': float(((1 + appreciation_rate * 1.5) ** target_years) - 1),
                    'trend': 'increasing'
                },
                'realistic': {
                    'rate': appreciation_rate,
                    'cost_per_sqm': float(target_cost),
                    'total_appreciation': float(total_appreciation),
                    'trend': 'increasing'
                },
                'conservative': {
                    'rate': max(-0.02, appreciation_rate * 0.3),  # Much lower, could be negative
                    'cost_per_sqm': float(adjusted_current_cost * ((1 + max(-0.02, appreciation_rate * 0.3)) ** target_years)),
                    'total_appreciation': float(((1 + max(-0.02, appreciation_rate * 0.3)) ** target_years) - 1),
                    'trend': 'increasing' if max(-0.02, appreciation_rate * 0.3) > 0 else 'decreasing'
                }
            }
        else:
            # Negative trend - scenarios around the rate
            scenarios = {
                'optimistic': {
                    'rate': min(0.02, appreciation_rate * 0.5),  # Less decrease, could be positive
                    'cost_per_sqm': float(adjusted_current_cost * ((1 + min(0.02, appreciation_rate * 0.5)) ** target_years)),
                    'total_appreciation': float(((1 + min(0.02, appreciation_rate * 0.5)) ** target_years) - 1),
                    'trend': 'increasing' if min(0.02, appreciation_rate * 0.5) > 0 else 'decreasing'
                },
                'realistic': {
                    'rate': appreciation_rate,
                    'cost_per_sqm': float(target_cost),
                    'total_appreciation': float(total_appreciation),
                    'trend': 'decreasing'
                },
                'conservative': {
                    'rate': appreciation_rate * 1.5,  # More decrease
                    'cost_per_sqm': float(adjusted_current_cost * ((1 + appreciation_rate * 1.5) ** target_years)),
                    'total_appreciation': float(((1 + appreciation_rate * 1.5) ** target_years) - 1),
                    'trend': 'decreasing'
                }
            }
        
        return {
            'current_prediction': {
                'year': current_year,
                'cost_per_sqm': float(adjusted_current_cost),
                'total_value': float(adjusted_current_cost * land_data.get('lot_area', 100)),
                'base_cost_per_sqm': float(current_cost),
                'location_adjustment': float(location_factor)
            },
            'future_prediction': {
                'target_year': current_year + target_years,
                'cost_per_sqm': float(target_cost),
                'total_value': float(target_cost * land_data.get('lot_area', 100)),
                'appreciation_rate': float(appreciation_rate),
                'total_appreciation': float(total_appreciation),
                'years_projected': target_years,
                'trend_direction': trend_direction,
                'is_increasing': is_increasing,
                'value_change': float(total_appreciation * 100)  # Percentage change
            },
            'yearly_breakdown': yearly_breakdown,
            'location_factors': {
                'location': location,
                'category': location_category,
                'multiplier': float(location_factor),
                'description': f"{location_category.title()} location ({'premium' if location_factor > 1.1 else 'discount' if location_factor < 0.9 else 'standard'} pricing)"
            },
            'scenarios': scenarios,
            'confidence': 'medium' if len(yearly_breakdown) > 0 else 'low'
        }
    
    def prepare_time_series_data(self, project_type=None, location=None, months_back=60):
        """
        Prepare time series data from historical land cost data for ARIMA forecasting
        
        Args:
            project_type: Filter by project type (optional)
            location: Filter by location (optional)
            months_back: Number of months of historical data to use
        
        Returns:
            pandas Series with monthly average cost per sqm, indexed by date
        """
        connection = self.connect_database()
        if not connection:
            return None
        
        try:
            query = """
                SELECT 
                    DATE_FORMAT(created_at, '%Y-%m-01') as month,
                    AVG(project_cost_numeric / NULLIF(lot_area, 0)) as avg_cost_per_sqm,
                    COUNT(*) as count
                FROM application_forms
                WHERE project_cost_numeric IS NOT NULL
                AND lot_area IS NOT NULL
                AND lot_area > 0
                AND created_at IS NOT NULL
                AND created_at >= DATE_SUB(CURDATE(), INTERVAL %s MONTH)
            """
            
            params = [months_back]
            
            if project_type:
                query += " AND project_type = %s"
                params.append(project_type)
            
            if location:
                query += " AND project_location = %s"
                params.append(location)
            
            query += " GROUP BY DATE_FORMAT(created_at, '%Y-%m-01') HAVING count >= 3 ORDER BY month"
            
            df = pd.read_sql(query, connection, params=params)
            connection.close()
            
            if len(df) < 12:  # Need at least 12 months for ARIMA
                return None
            
            # Convert to time series
            df['month'] = pd.to_datetime(df['month'])
            df.set_index('month', inplace=True)
            ts = df['avg_cost_per_sqm']
            
            # Fill any missing months with forward fill
            date_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='MS')
            ts = ts.reindex(date_range, method='ffill')
            
            return ts
            
        except Exception as e:
            print(f"Error preparing time series data: {e}")
            if connection:
                connection.close()
            return None
    
    def check_stationarity(self, ts):
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        if not ARIMA_AVAILABLE:
            return False
        
        try:
            result = adfuller(ts.dropna())
            return result[1] <= 0.05  # p-value <= 0.05 means stationary
        except:
            return False
    
    def make_stationary(self, ts, max_diff=2):
        """Make time series stationary by differencing"""
        ts_diff = ts.copy()
        d = 0
        
        for i in range(max_diff):
            if self.check_stationarity(ts_diff):
                break
            ts_diff = ts_diff.diff().dropna()
            d += 1
        
        return ts_diff, d
    
    def train_arima_model(self, ts, order=None, seasonal_order=None, auto_select=True):
        """
        Train ARIMA/SARIMAX model for land cost forecasting
        
        Args:
            ts: Time series data (pandas Series)
            order: ARIMA order (p, d, q) - if None, will auto-select
            seasonal_order: Seasonal order (P, D, Q, s) - if None, will use (1,1,1,12)
            auto_select: If True, automatically select best order
        
        Returns:
            Trained model and metadata
        """
        if not ARIMA_AVAILABLE:
            return None, {'error': 'ARIMA not available - statsmodels not installed'}
        
        if ts is None or len(ts) < 12:
            return None, {'error': 'Insufficient time series data (need at least 12 months)'}
        
        try:
            # Remove any NaN values
            ts_clean = ts.dropna()
            
            if len(ts_clean) < 12:
                return None, {'error': 'Insufficient data after cleaning'}
            
            # Auto-select order if requested
            if auto_select and order is None:
                # Try different orders and select best AIC
                best_aic = np.inf
                best_order = (1, 1, 1)
                best_seasonal = (1, 1, 1, 12)
                best_model = None
                
                # Try common ARIMA orders
                orders = [
                    (1, 1, 1), (1, 1, 0), (2, 1, 1), (1, 2, 1),
                    (0, 1, 1), (1, 0, 1), (2, 1, 0)
                ]
                
                for p, d, q in orders:
                    try:
                        # Try SARIMAX with seasonal component
                        model = SARIMAX(ts_clean, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                        fitted = model.fit(disp=False, maxiter=100)
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_seasonal = (1, 1, 1, 12)
                            best_model = fitted
                    except:
                        continue
                
                if best_model is None:
                    # Fallback to simple ARIMA
                    model = ARIMA(ts_clean, order=(1, 1, 1))
                    fitted = model.fit(disp=False, maxiter=100)
                    best_order = (1, 1, 1)
                    best_seasonal = None
                    best_model = fitted
                
                order = best_order
                seasonal_order = best_seasonal
            else:
                # Use provided or default orders
                if order is None:
                    order = (1, 1, 1)
                if seasonal_order is None:
                    seasonal_order = (1, 1, 1, 12)
                
                # Try SARIMAX first (with seasonal)
                try:
                    model = SARIMAX(ts_clean, order=order, seasonal_order=seasonal_order)
                    fitted = model.fit(disp=False, maxiter=100)
                except:
                    # Fallback to ARIMA (no seasonal)
                    model = ARIMA(ts_clean, order=order)
                    fitted = model.fit(disp=False, maxiter=100)
                    seasonal_order = None
            
            # Calculate model metrics
            residuals = fitted.resid
            mse = np.mean(residuals**2)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(mse)
            
            metadata = {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': float(fitted.aic),
                'bic': float(fitted.bic) if hasattr(fitted, 'bic') else None,
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'data_points': len(ts_clean),
                'model_type': 'SARIMAX' if seasonal_order else 'ARIMA'
            }
            
            return fitted, metadata
            
        except Exception as e:
            return None, {'error': f'ARIMA training failed: {str(e)}'}
    
    def forecast_arima(self, model, steps=12, confidence_level=0.95):
        """
        Generate forecast using trained ARIMA model
        
        Args:
            model: Trained ARIMA/SARIMAX model
            steps: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
        
        Returns:
            Dictionary with forecast, confidence intervals, and dates
        """
        if not ARIMA_AVAILABLE or model is None:
            return None
        
        try:
            # Generate forecast
            forecast = model.forecast(steps=steps)
            forecast_conf_int = model.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            # Generate future dates
            last_date = model.model.nobs  # Last observation index
            if hasattr(model.model, 'dates'):
                last_date_obj = model.model.dates[-1]
                future_dates = pd.date_range(start=last_date_obj, periods=steps+1, freq='MS')[1:]
            else:
                # Fallback: use current date
                future_dates = pd.date_range(start=pd.Timestamp.now(), periods=steps, freq='MS')
            
            # Prepare results
            forecast_data = []
            for i, date in enumerate(future_dates):
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_cost_per_sqm': float(forecast.iloc[i]) if hasattr(forecast, 'iloc') else float(forecast[i]),
                    'lower_bound': float(forecast_conf_int.iloc[i, 0]) if forecast_conf_int is not None else None,
                    'upper_bound': float(forecast_conf_int.iloc[i, 1]) if forecast_conf_int is not None else None
                })
            
            return {
                'forecast': forecast_data,
                'mean_forecast': float(forecast.mean()) if hasattr(forecast, 'mean') else float(np.mean(forecast)),
                'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates]
            }
            
        except Exception as e:
            print(f"Error in ARIMA forecast: {e}")
            return None
    
    def predict_land_cost_arima(self, land_data, forecast_months=12, project_type=None, location=None):
        """
        Predict land cost using ARIMA time series forecasting
        
        Args:
            land_data: Dict with land property details
            forecast_months: Number of months to forecast ahead
            project_type: Filter by project type (optional)
            location: Filter by location (optional)
        
        Returns:
            Dictionary with ARIMA-based predictions
        """
        if not ARIMA_AVAILABLE:
            return {
                'success': False,
                'error': 'ARIMA not available - statsmodels not installed',
                'fallback': 'Use predict_land_cost_future instead'
            }
        
        # Prepare time series data
        ts = self.prepare_time_series_data(project_type=project_type, location=location)
        
        if ts is None:
            return {
                'success': False,
                'error': 'Insufficient historical data for ARIMA forecasting (need at least 12 months)',
                'fallback': 'Use predict_land_cost_future instead'
            }
        
        # Train ARIMA model
        model, metadata = self.train_arima_model(ts, auto_select=True)
        
        if model is None:
            return {
                'success': False,
                'error': metadata.get('error', 'ARIMA model training failed'),
                'fallback': 'Use predict_land_cost_future instead'
            }
        
        # Generate forecast
        forecast_result = self.forecast_arima(model, steps=forecast_months)
        
        if forecast_result is None:
            return {
                'success': False,
                'error': 'ARIMA forecast generation failed',
                'fallback': 'Use predict_land_cost_future instead'
            }
        
        # Get current prediction (from ML model if available)
        current_prediction = self.predict_land_cost(land_data)
        current_cost = current_prediction.get('predicted_cost_per_sqm', 0) if current_prediction else 0
        
        # Get latest historical value
        latest_historical = float(ts.iloc[-1]) if len(ts) > 0 else current_cost
        
        # Calculate target year prediction
        target_month_idx = min(forecast_months - 1, len(forecast_result['forecast']) - 1)
        if target_month_idx >= 0:
            target_cost = forecast_result['forecast'][target_month_idx]['predicted_cost_per_sqm']
        else:
            target_cost = latest_historical
        
        # Calculate appreciation rate from ARIMA forecast
        if latest_historical > 0:
            appreciation_rate = (target_cost / latest_historical) ** (12.0 / forecast_months) - 1
        else:
            appreciation_rate = 0.03  # Default 3%
        
        # Prepare yearly breakdown
        yearly_breakdown = []
        current_year = datetime.now().year
        lot_area = land_data.get('lot_area', 100)
        
        for i, forecast_point in enumerate(forecast_result['forecast']):
            forecast_date = datetime.strptime(forecast_point['date'], '%Y-%m-%d')
            year = forecast_date.year
            month = forecast_date.month
            
            if not yearly_breakdown or yearly_breakdown[-1]['year'] != year:
                yearly_breakdown.append({
                    'year': year,
                    'month': month,
                    'cost_per_sqm': forecast_point['predicted_cost_per_sqm'],
                    'total_value': forecast_point['predicted_cost_per_sqm'] * lot_area,
                    'lower_bound': forecast_point.get('lower_bound'),
                    'upper_bound': forecast_point.get('upper_bound')
                })
        
        return {
            'success': True,
            'method': 'ARIMA',
            'current_prediction': {
                'year': current_year,
                'cost_per_sqm': float(latest_historical),
                'total_value': float(latest_historical * lot_area),
                'ml_prediction': float(current_cost) if current_cost > 0 else None
            },
            'future_prediction': {
                'target_year': current_year + (forecast_months // 12),
                'target_month': forecast_months,
                'cost_per_sqm': float(target_cost),
                'total_value': float(target_cost * lot_area),
                'appreciation_rate': float(appreciation_rate),
                'total_appreciation': float((target_cost / latest_historical) - 1) if latest_historical > 0 else 0,
                'is_increasing': appreciation_rate > 0
            },
            'monthly_forecast': forecast_result['forecast'],
            'yearly_breakdown': yearly_breakdown,
            'model_metadata': metadata,
            'confidence': 'high' if len(ts) >= 24 else 'medium' if len(ts) >= 12 else 'low'
        }
    
    def train_all_models(self):
        """Train all land prediction models"""
        print("=" * 60)
        print("Training Land Prediction Models")
        print("=" * 60)
        
        results = {}
        
        # Train land cost model
        print("\n[1/2] Training Land Cost Prediction Model...")
        # Use Random Forest for better accuracy and realism
        cost_results = self.train_land_cost_model('random_forest')
        
        if cost_results:
            if 'error' in cost_results:
                results['error'] = cost_results['error']
                print(f"Training error: {cost_results['error']}")
            else:
                results['land_cost'] = cost_results
                print(f"Training successful: {cost_results}")
        else:
            results['error'] = 'Training returned no results - check database has sufficient data'
            print("Training returned no results")
        
        # Save models if training was successful
        if 'land_cost' in results:
            print("\nSaving models...")
            try:
                self.save_models()
                self.save_model_metadata()
                results['models_saved'] = True
            except Exception as e:
                results['save_error'] = str(e)
                print(f"Error saving models: {e}")
        else:
            print("\nSkipping model save - training failed")
            results['models_saved'] = False
        
        print("\n" + "=" * 60)
        print("Model Training Completed!")
        print("=" * 60)
        
        return results


# Main execution
if __name__ == "__main__":
    db_config = {
        'host': 'srv1322.hstgr.io',
        'user': 'u520834156_uPAHOZone25',
        'password': 'Y+;a+*1y',
        'database': 'u520834156_dbUPAHOZoning'
    }
    
    land_predictions = LandPredictions(db_config)
    results = land_predictions.train_all_models()
    print(json.dumps(results, indent=2))

