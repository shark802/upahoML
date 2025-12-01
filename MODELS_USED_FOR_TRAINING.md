# Models Used for Training - Complete Guide

## Overview

The UPAHO Land Cost Prediction System uses multiple machine learning models for different prediction tasks. This document details all models, their purposes, algorithms, and training parameters.

---

## 🏠 Land Cost Prediction Models

### Primary Model: Linear Regression

**File:** `land_predictions.py`  
**Function:** `train_land_cost_model()`

#### Model Types Available:

1. **Linear Regression** (Default)
   - **Algorithm:** `sklearn.linear_model.LinearRegression`
   - **Formula:** `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`
   - **Use Case:** Base cost prediction per square meter
   - **Features:** 6 input features
   - **Training Data:** Historical land transactions from database

2. **Ridge Regression** (Optional)
   - **Algorithm:** `sklearn.linear_model.Ridge`
   - **Alpha:** 1.0 (regularization parameter)
   - **Use Case:** When dealing with multicollinearity
   - **Benefit:** Reduces overfitting

3. **Lasso Regression** (Optional)
   - **Algorithm:** `sklearn.linear_model.Lasso`
   - **Alpha:** 1.0 (regularization parameter)
   - **Use Case:** Feature selection and sparse models
   - **Benefit:** Can zero out less important features

4. **Polynomial Regression** (Optional)
   - **Algorithm:** `sklearn.preprocessing.PolynomialFeatures` + `LinearRegression`
   - **Degree:** 2 (quadratic features)
   - **Use Case:** Capturing non-linear relationships
   - **Benefit:** Can model curved relationships

#### Training Features:

| Feature | Type | Description |
|---------|------|-------------|
| `lot_area` | Numeric | Total lot size in square meters |
| `project_area` | Numeric | Building/development area in sqm |
| `year` | Numeric | Current year (for inflation) |
| `month` | Numeric | Current month (seasonal variations) |
| `age` | Numeric | Applicant age |
| `project_type_encoded` | Categorical (encoded) | Project type (residential, commercial, etc.) |

#### Target Variable:
- **Cost per square meter** = `project_cost_numeric / lot_area`

#### Training Process:

1. **Data Loading:**
   - Loads up to 5,000 records from `application_forms` table
   - Joins with `clients` table for age data

2. **Preprocessing:**
   - Handles missing values (median imputation)
   - Encodes categorical variables (LabelEncoder)
   - Removes outliers using IQR method (1.5 × IQR)
   - Feature scaling (StandardScaler)

3. **Train/Test Split:**
   - 80% training, 20% testing
   - Random state: 42 (for reproducibility)

4. **Model Training:**
   - Fits model on training data
   - Evaluates on test data

5. **Evaluation Metrics:**
   - **MAE** (Mean Absolute Error)
   - **MSE** (Mean Squared Error)
   - **RMSE** (Root Mean Squared Error)
   - **R² Score** (Coefficient of Determination)

#### Model Performance (Typical):
- **R² Score:** ~0.15-0.20 (moderate predictive power)
- **RMSE:** ~15,000-17,000 PHP/sqm
- **MAE:** ~13,000-14,000 PHP/sqm
- **Training Samples:** 3,000+ records

---

## 📊 Time Series Forecasting Models

### SARIMAX Model

**File:** `sarimax_predictions.py`  
**Function:** `train_sarimax_model()`

#### Algorithm:
- **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
- **Library:** `statsmodels.tsa.statespace.sarimax.SARIMAX`

#### Model Parameters:

**ARIMA Order (p, d, q):**
- `p`: Autoregressive order (typically 1-3)
- `d`: Differencing order (typically 1-2)
- `q`: Moving average order (typically 1-3)

**Seasonal Order (P, D, Q, s):**
- `P`: Seasonal autoregressive order (typically 1-2)
- `D`: Seasonal differencing order (typically 1)
- `Q`: Seasonal moving average order (typically 1-2)
- `s`: Seasonal period (12 for monthly data)

#### Auto-Selection:
- Uses `pmdarima.auto_arima` if available
- Automatically finds best (p,d,q)(P,D,Q,s) parameters
- Falls back to (1,1,1)(1,1,1,12) if not available

#### Use Case:
- Predicting future application counts by project type
- Monthly forecasts (12+ months ahead)
- Trend analysis and seasonality detection

#### Training Data:
- Time series data from `time_series_data` table
- Minimum 12 months of data required
- Supports exogenous variables (optional)

---

## 🎯 Application Approval Prediction Models

**File:** `ml_predictive_analytics.py`  
**Function:** `train_approval_prediction_model()`

### Available Model Types:

#### 1. XGBoost Classifier (Default)
- **Algorithm:** `xgboost.XGBClassifier`
- **Parameters:**
  - `n_estimators`: 200
  - `max_depth`: 8
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `random_state`: 42
  - `eval_metric`: 'logloss'

#### 2. Gradient Boosting Classifier
- **Algorithm:** `sklearn.ensemble.GradientBoostingClassifier`
- **Parameters:**
  - `n_estimators`: 200
  - `max_depth`: 8
  - `learning_rate`: 0.1
  - `random_state`: 42

#### 3. Random Forest Classifier
- **Algorithm:** `sklearn.ensemble.RandomForestClassifier`
- **Parameters:**
  - `n_estimators`: 200
  - `max_depth`: 10
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
  - `random_state`: 42
  - `class_weight`: 'balanced'

#### 4. Ensemble Model (Voting Classifier)
- **Algorithm:** `sklearn.ensemble.VotingClassifier`
- **Combines:**
  - Random Forest
  - Gradient Boosting
  - XGBoost (if available)
- **Voting:** 'soft' (probability-based)

#### Use Case:
- Predicting application approval probability
- Binary classification (approved/rejected)

#### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

## ⏱️ Processing Time Prediction Models

**File:** `ml_predictive_analytics.py`  
**Function:** `train_processing_time_model()`

### Available Model Types:

#### 1. XGBoost Regressor (Default)
- **Algorithm:** `xgboost.XGBRegressor`
- **Parameters:**
  - `n_estimators`: 200
  - `max_depth`: 8
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8

#### 2. Gradient Boosting Regressor
- **Algorithm:** `sklearn.ensemble.GradientBoostingRegressor`
- **Parameters:**
  - `n_estimators`: 200
  - `max_depth`: 8
  - `learning_rate`: 0.1

#### 3. Random Forest Regressor
- **Algorithm:** `sklearn.ensemble.RandomForestRegressor`
- **Parameters:**
  - `n_estimators`: 200
  - `max_depth`: 10
  - `min_samples_split`: 5

#### Use Case:
- Predicting application processing time
- Regression task (continuous output)

#### Evaluation Metrics:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 📈 Time Series Models (Advanced)

**File:** `ml_predictive_analytics.py`  
**Function:** `train_time_series_model()`

### Available Model Types:

#### 1. Prophet (Facebook)
- **Library:** `prophet.Prophet`
- **Use Case:** Long-term forecasting with seasonality
- **Features:**
  - Handles holidays
  - Trend detection
  - Seasonality (daily, weekly, yearly)

#### 2. ARIMA
- **Library:** `statsmodels.tsa.arima.model.ARIMA`
- **Order:** (1, 1, 1) by default
- **Use Case:** Short to medium-term forecasting

#### 3. Linear Trend (Fallback)
- Simple linear regression on time
- Used when advanced models unavailable

---

## 🔧 Model Training Configuration

### Data Preprocessing:

1. **Feature Scaling:**
   - **StandardScaler:** Normalizes features (mean=0, std=1)
   - Applied to all numeric features

2. **Categorical Encoding:**
   - **LabelEncoder:** Converts categories to integers
   - Used for project_type and other categorical variables

3. **Outlier Removal:**
   - **IQR Method:** Removes values outside Q1-1.5×IQR to Q3+1.5×IQR
   - Applied to cost per sqm values

4. **Missing Value Handling:**
   - **Median Imputation:** For numeric features
   - **Mode Imputation:** For categorical features

### Model Persistence:

Models are saved using `joblib`:
- **Model files:** `.pkl` format
- **Location:** `models/` directory
- **Files saved:**
  - `land_cost_model.pkl` - Main prediction model
  - `land_cost_scaler.pkl` - Feature scaler
  - `project_type_encoder.pkl` - Categorical encoder
  - `land_feature_importance.json` - Feature importance scores

### Model Metadata:

Stored in database (`model_performance` table):
- Model name and type
- Training date
- Performance metrics (MAE, RMSE, R²)
- Training/test sample counts
- Feature importance

---

## 📋 Summary Table

| Model | Algorithm | Type | Use Case | File |
|-------|-----------|------|----------|------|
| **Land Cost** | Linear Regression | Regression | Predict cost per sqm | `land_predictions.py` |
| **Land Cost (Ridge)** | Ridge Regression | Regression | Cost prediction (regularized) | `land_predictions.py` |
| **Land Cost (Lasso)** | Lasso Regression | Regression | Cost prediction (feature selection) | `land_predictions.py` |
| **Land Cost (Poly)** | Polynomial Regression | Regression | Non-linear cost relationships | `land_predictions.py` |
| **Time Series** | SARIMAX | Time Series | Future application trends | `sarimax_predictions.py` |
| **Approval** | XGBoost/RF/GB | Classification | Approval probability | `ml_predictive_analytics.py` |
| **Processing Time** | XGBoost/RF/GB | Regression | Processing duration | `ml_predictive_analytics.py` |
| **Forecasting** | Prophet/ARIMA | Time Series | Long-term forecasts | `ml_predictive_analytics.py` |

---

## 🎓 Model Selection Criteria

### When to Use Each Model:

**Linear Regression:**
- ✅ Simple, interpretable
- ✅ Fast training and prediction
- ✅ Good baseline model
- ✅ Works well with linear relationships

**Ridge/Lasso Regression:**
- ✅ When features are correlated (multicollinearity)
- ✅ Need regularization to prevent overfitting
- ✅ Lasso for feature selection

**Polynomial Regression:**
- ✅ When relationships are non-linear
- ✅ Need to capture curved patterns
- ⚠️ Can overfit with high degree

**XGBoost:**
- ✅ Best performance for complex patterns
- ✅ Handles non-linear relationships
- ✅ Feature importance available
- ⚠️ Requires more data

**Random Forest:**
- ✅ Robust to outliers
- ✅ Handles non-linear relationships
- ✅ Feature importance available
- ✅ Good default choice

**SARIMAX:**
- ✅ Time series with seasonality
- ✅ Monthly/quarterly patterns
- ✅ Can include external factors
- ⚠️ Requires sufficient historical data

---

## 📊 Training Data Requirements

### Minimum Data Requirements:

| Model | Minimum Records | Optimal Records |
|-------|----------------|-----------------|
| Linear Regression | 50-100 | 1,000+ |
| XGBoost/Random Forest | 200+ | 5,000+ |
| SARIMAX | 12 months | 24+ months |
| Ensemble Models | 500+ | 10,000+ |

### Data Quality Requirements:

1. **Completeness:**
   - No missing values in key features
   - Complete target variable (cost per sqm)

2. **Accuracy:**
   - Valid cost values (> 0)
   - Valid area values (> 0)
   - Realistic ranges

3. **Diversity:**
   - Multiple project types
   - Various locations
   - Different time periods

---

## 🔄 Model Retraining

### When to Retrain:

1. **Regular Schedule:**
   - Quarterly (every 3 months)
   - When new data accumulates (500+ new records)

2. **Performance Degradation:**
   - R² score drops significantly
   - Prediction errors increase
   - User feedback indicates issues

3. **Data Changes:**
   - New project types added
   - New locations added
   - Market conditions change

### Retraining Process:

1. Load new data from database
2. Preprocess (same as initial training)
3. Train model on updated dataset
4. Evaluate performance
5. Compare with previous model
6. Save if performance improved
7. Update metadata in database

---

## 📚 Libraries Used

### Core ML Libraries:
- **scikit-learn:** Linear models, ensemble methods, preprocessing
- **XGBoost:** Gradient boosting (optional)
- **statsmodels:** ARIMA, SARIMAX
- **Prophet:** Time series forecasting (optional)
- **pmdarima:** Auto ARIMA selection (optional)

### Data Processing:
- **pandas:** Data manipulation
- **numpy:** Numerical operations
- **joblib:** Model persistence

---

## 🎯 Current Model Status

**Active Models:**
- ✅ Linear Regression (Land Cost) - Primary model
- ✅ SARIMAX (Time Series) - For forecasting
- ⚠️ XGBoost (Optional) - If library installed
- ⚠️ Prophet (Optional) - If library installed

**Model Location:**
- Saved in: `models/` directory
- Format: `.pkl` files (joblib)
- Metadata: Database + JSON files

---

## 📝 Notes

1. **Model Selection:** Linear Regression is the default and most reliable for land cost prediction
2. **Optional Models:** XGBoost and Prophet require additional libraries
3. **Performance:** R² of 0.15-0.20 is reasonable for real estate (highly variable market)
4. **Regular Updates:** Models should be retrained quarterly for best accuracy
5. **Data Quality:** Model performance depends heavily on data quality and completeness

---

**Last Updated:** 2025  
**Model Version:** 1.0  
**Training Framework:** scikit-learn, statsmodels, XGBoost (optional)

