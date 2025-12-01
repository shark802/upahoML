# Quick Start - Train Model with Enhanced Mock Data

## 🚀 One-Command Training

Run this single command to:
1. Generate 2000 realistic mock records with ALL feature engineering features
2. Insert into database
3. Train the Random Forest model

```bash
python train_with_mock_data.py
```

---

## ✅ What Gets Generated

### Enhanced Mock Data Includes:

**Basic Features:**
- ✅ Lot area, project area
- ✅ Project type, location
- ✅ Year, month, age

**NEW - Feature Engineering Features:**
- ✅ **Latitude/Longitude** - For distance calculations
- ✅ **Site Zoning** - For zoning categories
- ✅ **Location Type** - For location classification
- ✅ **Temporal Features** - Day of week, quarter, seasonality
- ✅ **Size Features** - Ratios, categories, interactions
- ✅ **Location Features** - Distance to center, categories
- ✅ **Interaction Features** - Size×Location, Type×Size, Year×Location

**Total: 20+ Features** (up from 6)

---

## 📊 Expected Results

After training, you should see:

```
Model Type: Random Forest Regressor
Training Samples: ~1600
Test Samples: ~400

Performance Metrics:
  R² Score: 0.40-0.70 (expected)
  RMSE: 8,000-12,000 PHP/sqm
  MAE: 7,000-10,000 PHP/sqm
```

---

## 🔧 Manual Steps (If Needed)

### Option 1: Via Python Script
```bash
python train_with_mock_data.py
```

### Option 2: Via API
```bash
# Step 1: Insert mock data
POST /api/insert_training_data?num_records=2000

# Step 2: Train model
POST /api/train
```

### Option 3: Step by Step
```python
from generate_mock_data import *
from land_predictions import LandPredictions

# 1. Generate data
db_config = get_db_config()
connection = connect_database(db_config)
data = generate_mock_data(2000)
insert_mock_data(connection, data)

# 2. Train model
lp = LandPredictions(db_config)
results = lp.train_all_models()
```

---

## 📋 What the Script Does

1. **Connects to Database**
   - Uses config.json or environment variables
   - Verifies connection

2. **Generates Enhanced Mock Data**
   - 2000 records with realistic distributions
   - Includes ALL feature engineering features
   - Proper location coordinates
   - Zoning classifications
   - Temporal diversity (3 years)

3. **Inserts Data**
   - Batch processing (50 records at a time)
   - Creates clients and applications
   - Handles duplicates gracefully

4. **Trains Model**
   - Random Forest Regressor
   - Uses all 20+ engineered features
   - Evaluates performance
   - Saves models

---

## 🎯 Features Generated

### Location Features:
- **Coordinates:** Realistic lat/lon based on location type
- **Distance:** Calculated to city center
- **Categories:** Premium/Standard/Economy encoding

### Zoning Features:
- **Site Zoning:** Matched to project type
- **Categories:** Premium/Standard/Economy encoding

### Temporal Features:
- **3 Years of Data:** 2022-2025
- **Seasonal Variation:** All months represented
- **Day of Week:** Full week coverage

### Size Features:
- **Diverse Sizes:** 80 sqm to 10,000 sqm
- **Realistic Ratios:** Project area vs lot area
- **Categories:** Small/Medium/Large

---

## 📈 Performance Expectations

### Before (6 Features, Linear Regression):
- R²: 0.15-0.20
- RMSE: 15-17k PHP/sqm

### After (20+ Features, Random Forest):
- R²: **0.40-0.70** ✅
- RMSE: **8-12k PHP/sqm** ✅
- **2-3x better accuracy**

---

## 🔍 Verify Training

### Check Model Files:
```bash
ls models/
# Should see:
# - land_cost_model.pkl
# - land_cost_scaler.pkl
# - project_type_encoder.pkl
# - land_feature_importance.json
```

### Test Prediction:
```python
from land_predictions import LandPredictions

lp = LandPredictions(db_config)
lp.load_models()

prediction = lp.predict_land_cost({
    'lot_area': 200,
    'project_area': 150,
    'project_type': 'residential',
    'location': 'Downtown',
    'latitude': 14.5995,
    'longitude': 120.9842,
    'year': 2024,
    'month': 1,
    'age': 35
})

print(prediction)
```

---

## ⚠️ Troubleshooting

### Issue: "No data available"
**Solution:** Check database connection and ensure data was inserted

### Issue: "Training failed"
**Solution:** 
- Check database has records: `SELECT COUNT(*) FROM application_forms`
- Verify all required fields exist
- Check error logs

### Issue: "Low R² score"
**Solution:**
- Add more training data (increase num_records)
- Check data quality
- Verify feature engineering is working

---

## 📝 Notes

- **Coordinates:** Currently set for Philippines (Manila area)
  - Update in `generate_mock_data.py` if needed
- **Data Volume:** 2000 records is good, more is better
- **Training Time:** 1-3 minutes depending on data size
- **Model Type:** Random Forest (best for realistic predictions)

---

**Ready to train? Run:**
```bash
python train_with_mock_data.py
```

