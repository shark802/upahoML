# Testing Page Guide - Land Property Value Prediction

## 🧪 Testing Page Created

A comprehensive testing page has been created: **`test_predictions.html`**

---

## 🚀 How to Use

### Step 1: Open the Testing Page

**Local (XAMPP):**
```
http://localhost/upahoML/test_predictions.html
```

**Online:**
```
https://upahozoning.bccbsis.com/test_predictions.html
```

### Step 2: Fill in the Form

The form includes all feature engineering inputs:

**Required Fields:**
- ✅ Lot Area (sqm)
- ✅ Project Area (sqm)
- ✅ Project Type
- ✅ Location

**Optional (but Recommended):**
- ⚠️ Latitude/Longitude (for distance calculations)
- ⚠️ Site Zoning (for zoning categories)
- ⚠️ Location Type (for location classification)

**Temporal Fields:**
- Year, Month, Age
- Target Years (5 or 10)

### Step 3: Quick Fill Options

**"Fill Sample Data" Button:**
- Fills form with realistic sample data
- Premium location example
- Ready to test immediately

**"Use City Center" Button:**
- Fills coordinates with city center
- Quick way to test distance features

### Step 4: Submit and View Results

Click **"🔮 Predict Land Property Value"** to:
- Send data to API
- Get predictions
- View detailed results

---

## 📊 What the Testing Page Shows

### 1. **Model Information**
- Model type (Random Forest)
- Number of features used
- R² Score (accuracy)

### 2. **Current Year Prediction**
- Cost per square meter
- Total land value
- Location category and multiplier

### 3. **Future Prediction (5-10 Years)**
- Future cost per sqm
- Future total value
- Yearly appreciation/depreciation rate
- Total change percentage
- Trend indicator (increasing/decreasing)

### 4. **Three Scenarios**
- **Optimistic:** Best-case scenario
- **Realistic:** Most likely outcome
- **Conservative:** Worst-case scenario

### 5. **Yearly Breakdown**
- Year-by-year progression
- Cost per sqm for each year
- Percentage change from current

### 6. **Features Used**
- List of all features used in prediction
- Shows which engineered features are active

---

## 🎯 Testing Scenarios

### Scenario 1: Premium Location
```
Lot Area: 200 sqm
Project Area: 150 sqm
Project Type: Residential
Location: Downtown
Latitude: 14.5995
Longitude: 120.9842
Site Zoning: Residential-High
Location Type: Urban
```

**Expected:** Higher value due to premium location

### Scenario 2: Rural Area
```
Lot Area: 500 sqm
Project Area: 300 sqm
Project Type: Agricultural
Location: Rural
Latitude: 14.6500
Longitude: 121.0500
Site Zoning: Agricultural
Location Type: Rural
```

**Expected:** Lower value, possible depreciation

### Scenario 3: Commercial Downtown
```
Lot Area: 300 sqm
Project Area: 250 sqm
Project Type: Commercial
Location: Commercial District
Latitude: 14.5950
Longitude: 120.9800
Site Zoning: Commercial
Location Type: Urban
```

**Expected:** Very high value, strong appreciation

---

## 🔧 Features Tested

The testing page validates all feature engineering:

### ✅ Size Features
- Area ratio calculations
- Size categories
- Efficiency ratios

### ✅ Location Features
- Location category encoding
- Distance to city center (if coordinates provided)
- Location multipliers

### ✅ Temporal Features
- Year and month
- Seasonality (cyclical encoding)
- Day of week effects

### ✅ Zoning Features
- Zoning category encoding
- Project type classification

### ✅ Interaction Features
- Size × Location interactions
- Type × Size interactions
- Year × Location interactions

---

## 📱 Responsive Design

The page is fully responsive:
- ✅ Desktop: Side-by-side form and results
- ✅ Tablet: Stacked layout
- ✅ Mobile: Optimized for small screens

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to server"
**Solution:**
- Check `predict_land_cost_api.php` is in the same directory
- Verify API URL in PHP file is correct
- Check browser console (F12) for errors

### Issue: "Network error"
**Solution:**
- Verify PHP file is accessible
- Check server error logs
- Test API directly: `POST predict_land_cost_api.php`

### Issue: Results not showing
**Solution:**
- Open browser console (F12)
- Check for JavaScript errors
- Verify API response format

### Issue: Missing features
**Solution:**
- Optional features (lat/lon, zoning) improve accuracy but aren't required
- Model will use defaults if not provided

---

## 📊 Expected Results

### Good Prediction Example:
```
Current: 43,100 PHP/sqm
Future (10 years): 58,745 PHP/sqm
Appreciation: +36.3% over 10 years
Rate: +3.14% per year
```

### Model Performance:
- **R² Score:** 0.5336 (53.4% accuracy)
- **RMSE:** ±11,599 PHP/sqm
- **Confidence:** Medium to High

---

## 🎓 Testing Tips

1. **Test Different Locations:**
   - Compare Downtown vs Rural
   - See location premium effects

2. **Test Different Sizes:**
   - Small lots (< 100 sqm)
   - Large lots (> 500 sqm)
   - See size category effects

3. **Test Different Project Types:**
   - Residential vs Commercial
   - See type-specific pricing

4. **Test With/Without Coordinates:**
   - With coordinates: Distance calculations active
   - Without coordinates: Uses location category only

5. **Test Different Years:**
   - Current year vs future years
   - See temporal trends

---

## 📝 API Testing

You can also test the API directly:

### Using curl:
```bash
curl -X POST http://localhost/upahoML/predict_land_cost_api.php \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_type": "land_cost_future",
    "target_years": 10,
    "data": {
      "lot_area": 200,
      "project_area": 150,
      "project_type": "residential",
      "location": "Downtown",
      "latitude": 14.5995,
      "longitude": 120.9842,
      "year": 2024,
      "month": 12,
      "age": 35
    }
  }'
```

### Using JavaScript (Browser Console):
```javascript
fetch('predict_land_cost_api.php', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prediction_type: 'land_cost_future',
    target_years: 10,
    data: {
      lot_area: 200,
      project_area: 150,
      project_type: 'residential',
      location: 'Downtown',
      latitude: 14.5995,
      longitude: 120.9842,
      year: 2024,
      month: 12,
      age: 35
    }
  })
})
.then(r => r.json())
.then(console.log);
```

---

## ✅ Quick Start

1. **Open:** `test_predictions.html` in browser
2. **Click:** "Fill Sample Data" button
3. **Click:** "🔮 Predict Land Property Value"
4. **View:** Results with all predictions

---

## 🎯 What to Look For

### Good Signs:
- ✅ Predictions are realistic (not extreme)
- ✅ Location premium is applied correctly
- ✅ Future predictions show reasonable appreciation
- ✅ Scenarios show appropriate ranges
- ✅ Yearly breakdown shows smooth progression

### Red Flags:
- ⚠️ Predictions are too high/low
- ⚠️ No location premium applied
- ⚠️ Future predictions don't make sense
- ⚠️ Errors in console

---

**Testing Page:** `test_predictions.html`  
**Status:** ✅ Ready to Use  
**Features:** All 23 engineered features supported

