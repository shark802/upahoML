# Base Code Structure - Complete Overview

## 📁 Core Files

### 1. **`land_predictions.py`** - Main ML Model Class
**Purpose:** Contains the `LandPredictions` class that handles all land cost prediction logic.

#### Key Components:

**Class: `LandPredictions`**
- **Initialization:** Sets up database connection, model storage, encoders, scalers
- **Models Directory:** Creates/uses `models/` folder for saving trained models

**Main Methods:**

1. **`connect_database()`** (Lines 53-67)
   - Connects to MySQL database
   - Uses configuration from `db_config`
   - Returns connection object or None

2. **`load_land_data()`** (Lines 69-118)
   - Loads historical land transaction data from database
   - SQL Query joins `application_forms` and `clients` tables
   - Calculates `cost_per_sqm = project_cost_numeric / lot_area`
   - Returns pandas DataFrame with up to 5,000 records

3. **`preprocess_land_cost_data()`** (Lines 120-199)
   - Cleans and prepares data for training
   - Handles missing values (median imputation)
   - Encodes categorical variables (LabelEncoder for project_type)
   - Removes outliers using IQR method (1.5 × IQR)
   - Returns features (X), target (y), and feature list

4. **`train_land_cost_model()`** (Lines 201-298)
   - **Main training function**
   - Supports: 'linear', 'ridge', 'lasso', 'polynomial'
   - Splits data: 80% train, 20% test
   - Scales features using StandardScaler
   - Trains model and evaluates performance
   - Returns metrics: MAE, MSE, RMSE, R²

5. **`predict_land_cost()`** (Lines 300-357)
   - **Current year prediction**
   - Takes land_data dict as input
   - Prepares features in correct order
   - Encodes project_type
   - Scales features
   - Returns predicted cost per sqm

6. **`predict_land_cost_future()`** (Lines 720-882)
   - **Future prediction (5-10 years)**
   - Gets current year prediction
   - Calculates location factors
   - Calculates appreciation/depreciation rate
   - Projects future costs using compound growth
   - Creates optimistic/realistic/conservative scenarios
   - Returns complete prediction with yearly breakdown

7. **`calculate_location_factors()`** (Lines 588-635)
   - Analyzes historical data by location
   - Calculates location multipliers
   - Returns dict with location factors

8. **`calculate_yearly_appreciation_rate()`** (Lines 637-718)
   - Calculates realistic appreciation/depreciation rate
   - Uses year-over-year growth analysis
   - Returns rate (can be positive or negative)
   - Capped between -10% and +15%

9. **`save_models()`** (Lines 454-487)
   - Saves trained models to `.pkl` files
   - Saves encoders, scalers, feature importance

10. **`load_models()`** (Lines 489-531)
    - Loads saved models from disk
    - Restores encoders, scalers, feature importance

---

### 2. **`app.py`** - Flask API Server
**Purpose:** REST API that exposes the prediction models via HTTP endpoints.

#### Key Components:

**Flask App Setup:**
- Flask application with CORS enabled
- Lazy loading of models (initialized on first use)

**Configuration Functions:**

1. **`get_db_config()`** (Lines 26-46)
   - Reads database config from `config.json` or environment variables
   - Supports Heroku config vars (DB_HOST, DB_USER, etc.)

2. **`init_land_predictions()`** (Lines 48-122)
   - Initializes `LandPredictions` instance
   - Tests database connection
   - Auto-trains models if not found
   - Handles errors gracefully

3. **`init_ml_analytics()`** (Lines 124-136)
   - Initializes ML analytics models (optional)

**API Endpoints:**

1. **`GET /`** (Line 138)
   - Serves the HTML UI (`land_cost_prediction_ui.html`)

2. **`GET /api`** (Line 143)
   - API information endpoint
   - Lists available endpoints

3. **`POST /predict/land_cost`** (Lines 157-188)
   - Predicts current year land cost
   - Accepts JSON with land data
   - Returns prediction result

4. **`POST /predict/land_cost_future`** (Lines 190-224)
   - Predicts future land cost (5-10 years)
   - Accepts: `target_years`, `data`
   - Returns complete prediction with scenarios

5. **`POST /api/predict`** (Lines 226-269)
   - Universal prediction endpoint
   - Supports both `land_cost` and `land_cost_future`
   - Compatible with PHP format

6. **`POST /api/train`** (Lines 311-402)
   - Trains models from database
   - Checks database connection
   - Returns training results

7. **`GET /api/check`** (Lines 507-587)
   - Health check endpoint
   - Checks database connection
   - Checks model status

---

## 🔄 Data Flow

### Training Flow:
```
1. Database → load_land_data() → DataFrame
2. DataFrame → preprocess_land_cost_data() → (X, y)
3. (X, y) → train_test_split() → (X_train, X_test, y_train, y_test)
4. X_train → StandardScaler → X_train_scaled
5. X_train_scaled → LinearRegression.fit() → Trained Model
6. Model → save_models() → .pkl files
```

### Prediction Flow:
```
1. User Input (JSON) → API Endpoint
2. API → init_land_predictions() → LandPredictions instance
3. LandPredictions → load_models() → Load saved models
4. Input Data → predict_land_cost() → Base prediction
5. Base + Location → calculate_location_factors() → Adjusted cost
6. Adjusted + Rate → predict_land_cost_future() → Future prediction
7. Result → JSON Response → User
```

---

## 📊 Database Schema

### Tables Used:

**`application_forms`**
- `id` - Primary key
- `project_type` - Type of project (residential, commercial, etc.)
- `project_location` - Location name
- `project_area` - Building area in sqm
- `lot_area` - Lot size in sqm
- `project_cost_numeric` - Total project cost
- `created_at` - Date of application
- `client_id` - Foreign key to clients table

**`clients`**
- `id` - Primary key
- `age` - Applicant age
- `gender` - Applicant gender

**`model_performance`** (Optional)
- Stores model metadata and performance metrics

---

## 🔧 Configuration

### Database Config (`config.json`):
```json
{
  "host": "srv1322.hstgr.io",
  "user": "u520834156_uPAHOZone25",
  "password": "Y+;a+*1y",
  "database": "u520834156_dbUPAHOZoning",
  "port": 3306
}
```

### Environment Variables (Heroku):
- `DB_HOST` - Database hostname
- `DB_USER` - Database username
- `DB_PASSWORD` - Database password
- `DB_NAME` - Database name
- `DB_PORT` - Database port (default: 3306)
- `PORT` - Flask server port (default: 5000)

---

## 📦 Model Files Saved

When models are trained, these files are saved in `models/` directory:

1. **`land_cost_model.pkl`** - Trained Linear Regression model
2. **`land_cost_scaler.pkl`** - StandardScaler for feature scaling
3. **`project_type_encoder.pkl`** - LabelEncoder for project types
4. **`land_feature_importance.json`** - Feature importance scores
5. **`land_cost_poly.pkl`** - Polynomial transformer (if polynomial model used)

---

## 🎯 Key Algorithms

### 1. Linear Regression
- **Formula:** `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`
- **Features:** 6 inputs (lot_area, project_area, year, month, age, project_type_encoded)
- **Target:** Cost per square meter

### 2. Location Factor Calculation
- **Formula:** `Location_Factor = Avg_Cost_Location / Avg_Cost_Overall`
- **Categories:**
  - Premium: ≥ 1.2x
  - Standard: 0.9x - 1.2x
  - Economy: ≤ 0.85x

### 3. Appreciation Rate Calculation
- **Method:** Year-over-year growth analysis
- **Statistics:** Median (robust) or weighted average
- **Range:** -10% to +15% per year
- **Can be negative** (depreciation)

### 4. Future Cost Projection
- **Formula:** `Future_Cost = Current_Cost × (1 + Rate)^Years`
- **Works for:** Both positive (increase) and negative (decrease) rates
- **Minimum Floor:** 10% of original value

---

## 🔍 Feature Engineering

### Input Features:
1. **lot_area** - Numeric (square meters)
2. **project_area** - Numeric (square meters)
3. **year** - Numeric (extracted from created_at)
4. **month** - Numeric (1-12, extracted from created_at)
5. **age** - Numeric (applicant age)
6. **project_type_encoded** - Categorical (encoded to integer)

### Preprocessing Steps:
1. **Missing Value Handling:** Median imputation for numeric, mode for categorical
2. **Encoding:** LabelEncoder for project_type
3. **Scaling:** StandardScaler (mean=0, std=1)
4. **Outlier Removal:** IQR method (Q1 - 1.5×IQR to Q3 + 1.5×IQR)

---

## 🚀 API Request/Response Format

### Request (Current Year):
```json
{
  "prediction_type": "land_cost",
  "data": {
    "lot_area": 200,
    "project_area": 150,
    "project_type": "residential",
    "year": 2024,
    "month": 1,
    "age": 35
  }
}
```

### Response (Current Year):
```json
{
  "success": true,
  "prediction": {
    "predicted_cost_per_sqm": 34856.25,
    "confidence": "medium",
    "model_r2": 0.156
  }
}
```

### Request (Future):
```json
{
  "prediction_type": "land_cost_future",
  "target_years": 10,
  "data": {
    "lot_area": 200,
    "project_area": 150,
    "project_type": "residential",
    "location": "Downtown",
    "year": 2024,
    "month": 1,
    "age": 35
  }
}
```

### Response (Future):
```json
{
  "success": true,
  "prediction": {
    "current_prediction": {
      "year": 2024,
      "cost_per_sqm": 43100.0,
      "total_value": 8620000.0
    },
    "future_prediction": {
      "target_year": 2034,
      "cost_per_sqm": 58745.0,
      "total_value": 11749000.0,
      "appreciation_rate": 0.0314,
      "total_appreciation": 0.363,
      "is_increasing": true
    },
    "yearly_breakdown": [...],
    "location_factors": {...},
    "scenarios": {
      "optimistic": {...},
      "realistic": {...},
      "conservative": {...}
    }
  }
}
```

---

## 🛠️ Error Handling

### Database Errors:
- Connection timeout handling
- Localhost detection (won't work on Heroku)
- Connection error messages

### Model Errors:
- Model not found → Auto-train if enabled
- Training failure → Returns error message
- Prediction failure → Returns error with details

### Data Errors:
- Missing data → Returns error
- Invalid input → Returns validation error
- Preprocessing failure → Returns error

---

## 📝 Key Design Patterns

1. **Lazy Loading:** Models loaded only when needed
2. **Auto-Training:** Models train automatically if not found
3. **Error Recovery:** Graceful error handling with informative messages
4. **Model Persistence:** Save/load models to avoid retraining
5. **Configuration Flexibility:** Support both file and environment configs

---

## 🔗 File Dependencies

```
app.py
  └── land_predictions.py
      ├── sklearn (LinearRegression, StandardScaler, etc.)
      ├── pandas (DataFrame operations)
      ├── numpy (Numerical operations)
      ├── joblib (Model persistence)
      └── mysql.connector (Database connection)

land_predictions.py
  └── Database (MySQL)
      ├── application_forms table
      └── clients table
```

---

## 🎓 Usage Examples

### Training Models:
```python
from land_predictions import LandPredictions

db_config = {
    'host': 'your-host',
    'user': 'your-user',
    'password': 'your-password',
    'database': 'your-database'
}

lp = LandPredictions(db_config)
results = lp.train_all_models()
lp.save_models()
```

### Making Predictions:
```python
land_data = {
    'lot_area': 200,
    'project_area': 150,
    'project_type': 'residential',
    'location': 'Downtown',
    'year': 2024,
    'month': 1,
    'age': 35
}

# Current prediction
current = lp.predict_land_cost(land_data)

# Future prediction
future = lp.predict_land_cost_future(land_data, target_years=10)
```

---

## 📌 Important Notes

1. **Model Directory:** Models saved in `models/` folder (or `/tmp/models` on Heroku if read-only)
2. **Database Required:** System needs database connection for training and location factors
3. **Auto-Training:** Models auto-train on first API call if not found
4. **CORS Enabled:** API allows cross-origin requests
5. **Error Logging:** All errors logged to Flask logger

---

**Last Updated:** 2025  
**Code Version:** 1.0  
**Main Files:** `land_predictions.py`, `app.py`

