# Your API Endpoints Reference

## 🌐 Base URL
```
https://endpoint-upaho-3e0044d5e3a6.herokuapp.com
```

---

## 📍 Available Endpoints

### 1. **Current Year Prediction**
**Endpoint:** `POST /predict/land_cost`

**Full URL:**
```
https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/predict/land_cost
```

**Request Format:**
```json
{
  "lot_area": 200,
  "project_area": 150,
  "project_type": "residential",
  "location": "Downtown",
  "year": 2024,
  "month": 12,
  "age": 35
}
```

**Or using PHP-style format:**
```json
{
  "prediction_type": "land_cost",
  "data": {
    "lot_area": 200,
    "project_area": 150,
    "project_type": "residential",
    "location": "Downtown",
    "year": 2024,
    "month": 12,
    "age": 35
  }
}
```

---

### 2. **Future Prediction (5-10 Years)** ⭐ RECOMMENDED
**Endpoint:** `POST /predict/land_cost_future`

**Full URL:**
```
https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/predict/land_cost_future
```

**Request Format:**
```json
{
  "target_years": 10,
  "data": {
    "lot_area": 200,
    "project_area": 150,
    "project_type": "residential",
    "location": "Downtown",
    "year": 2024,
    "month": 12,
    "age": 35
  }
}
```

**Or using PHP-style format:**
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
    "month": 12,
    "age": 35
  }
}
```

**Optional fields you can add:**
```json
{
  "target_years": 10,
  "data": {
    "lot_area": 200,
    "project_area": 150,
    "project_type": "residential",
    "location": "Downtown",
    "latitude": 14.5995,
    "longitude": 120.9842,
    "site_zoning": "Residential",
    "location_type": "Urban",
    "year": 2024,
    "month": 12,
    "age": 35
  }
}
```

---

### 3. **Universal Endpoint** (PHP Compatible)
**Endpoint:** `POST /api/predict`

**Full URL:**
```
https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/api/predict
```

**Request Format:**
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
    "month": 12,
    "age": 35
  }
}
```

---

### 4. **API Information**
**Endpoint:** `GET /api`

**Full URL:**
```
https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/api
```

Returns information about available endpoints.

---

### 5. **Testing Page**
**Endpoint:** `GET /test`

**Full URL:**
```
https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/test
```

Opens a testing interface for predictions.

---

## 🔧 For PHP Integration

### Update `predict_land_cost_api.php`

Set your API URL on **line 64**:

```php
$api_url = 'https://endpoint-upaho-3e0044d5e3a6.herokuapp.com';
```

Or use environment variable:
```php
$api_url = getenv('HEROKU_API_URL') ?: 'https://endpoint-upaho-3e0044d5e3a6.herokuapp.com';
```

---

## 📝 Example cURL Commands

### Test Current Year Prediction
```bash
curl -X POST https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/predict/land_cost \
  -H "Content-Type: application/json" \
  -d '{
    "lot_area": 200,
    "project_area": 150,
    "project_type": "residential",
    "location": "Downtown",
    "year": 2024,
    "month": 12,
    "age": 35
  }'
```

### Test Future Prediction
```bash
curl -X POST https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/predict/land_cost_future \
  -H "Content-Type: application/json" \
  -d '{
    "target_years": 10,
    "data": {
      "lot_area": 200,
      "project_area": 150,
      "project_type": "residential",
      "location": "Downtown",
      "year": 2024,
      "month": 12,
      "age": 35
    }
  }'
```

---

## 📊 Response Format

**Success Response:**
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
      "total_appreciation": 0.363
    },
    "yearly_breakdown": [...],
    "location_factors": {...},
    "scenarios": {...}
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message here"
}
```

---

## ✅ Required Fields (Minimum)

```json
{
  "lot_area": 200,        // Required
  "project_area": 150,    // Required
  "project_type": "residential"  // Required
}
```

All other fields (location, year, month, age, etc.) are optional and will use defaults.

---

## 🔍 Quick Test

Visit in browser:
```
https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/api
```

This should return:
```json
{
  "status": "ok",
  "service": "UPAHO Land Cost Prediction API",
  "endpoints": {...}
}
```

---

## 🚀 Summary

**For most use cases, use this endpoint:**
```
POST https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/predict/land_cost_future
```

This gives you both current and future predictions with scenarios!


