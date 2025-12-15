# How to Test the Land Cost Prediction

After fixing the 21 vs 23 feature mismatch issue, here are several ways to test if the prediction is working:

## 🧪 Method 1: Python Test Script (Recommended)

Run the test script directly:

```bash
python test_prediction.py
```

This script will:
- ✅ Load models
- ✅ Check feature count (should be 23)
- ✅ Test current year prediction
- ✅ Test future prediction (10 years)
- ✅ Test with minimal data (missing optional fields)

**Expected Output:**
```
✅ Models loaded successfully!
✅ Scaler expects 23 features
✅ Current year prediction successful!
✅ Future prediction successful!
✅ ALL TESTS PASSED!
```

---

## 🌐 Method 2: Using the Testing HTML Page

1. **If testing locally:**
   - Open `test_predictions.html` in your browser
   - Or serve it: `python -m http.server 8000` then visit `http://localhost:8000/test_predictions.html`

2. **If testing on Heroku:**
   - Visit: `https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/test`
   - Or visit the root: `https://endpoint-upaho-3e0044d5e3a6.herokuapp.com/`

3. **Fill in the form and click "Predict Land Property Value"**

**What to look for:**
- ✅ No errors in browser console (F12)
- ✅ Prediction results display correctly
- ✅ Shows "23 features" in model information
- ✅ All prediction values are displayed

---

## 🔧 Method 3: Test via cURL (Command Line)

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

### Test Future Prediction (10 years)

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

**Expected Response:**
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
      ...
    },
    ...
  }
}
```

---

## 🐍 Method 4: Test Directly in Python

```python
from land_predictions import LandPredictions

# Initialize
db_config = {
    'host': 'your-host',
    'user': 'your-user',
    'password': 'your-password',
    'database': 'your-database'
}
lp = LandPredictions(db_config)

# Load models
lp.load_models()

# Test data
test_data = {
    'lot_area': 200,
    'project_area': 150,
    'project_type': 'residential',
    'location': 'Downtown',
    'year': 2024,
    'month': 12,
    'age': 35
}

# Test prediction
result = lp.predict_land_cost(test_data)
print(f"Predicted cost: {result['predicted_cost_per_sqm']} PHP/sqm")
print(f"Features used: {result['features_used']}")  # Should be 23

# Test future prediction
future_result = lp.predict_land_cost_future(test_data, target_years=10)
print(f"Future cost: {future_result['future_prediction']['cost_per_sqm']} PHP/sqm")
```

---

## ✅ Success Indicators

### ✅ Everything is Working If:

1. **No errors about feature count:**
   - ❌ Should NOT see: "X has 21 features, but StandardScaler is expecting 23"
   - ✅ Should see: "SUCCESS: Feature count matches: 23 features"

2. **Prediction returns results:**
   - ✅ `result['predicted_cost_per_sqm']` has a positive number
   - ✅ `result['features_used']` equals 23

3. **All features are generated:**
   - ✅ `distance_to_center` is created (even if lat/lon missing, defaults to 10.0)
   - ✅ `zoning_category` is created (even if site_zoning missing, defaults to 2)
   - ✅ `location_category` is created (even if location missing, defaults to 2)
   - ✅ All interaction features are created

4. **Model metadata is loaded:**
   - ✅ Feature list from training is available
   - ✅ Scaler expects exactly 23 features

---

## ❌ Troubleshooting

### Issue: "Models not loaded"
**Solution:**
```bash
# Train models first
python land_predictions.py
```

### Issue: "Still getting 21 features error"
**Solution:**
1. Check that you've saved the updated `land_predictions.py`
2. Restart the Flask app (if running)
3. Check Heroku logs: `heroku logs --tail`
4. Ensure all 23 features are always created in `_engineer_features()`

### Issue: "Prediction returns None"
**Solution:**
- Check database connection
- Verify models exist in `models/` directory
- Check Heroku logs for detailed errors

### Issue: "Network error in browser"
**Solution:**
- Check CORS headers are set
- Verify API endpoint URL is correct
- Check browser console (F12) for detailed errors

---

## 📊 What Was Fixed

The fix ensures that **all 23 features are always created**, even when optional input data is missing:

1. **`distance_to_center`**: Always created (default: 10.0 km if lat/lon missing)
2. **`zoning_category`**: Always created (default: 2 if site_zoning missing)
3. **`location_category`**: Always created (default: 2 if location missing)
4. **All interaction features**: Always created with defaults if components missing
5. **All temporal features**: Always created from year/month or defaults

This ensures the scaler always receives exactly 23 features as expected.

---

## 🔍 Debug Information

If you want to see detailed debug output, the code will print:
- `CRITICAL: Scaler expects X features`
- `SUCCESS: Feature count matches: X features`
- Warning messages if any features are missing/NaN

Check the logs (Heroku logs or console) to see these messages.


