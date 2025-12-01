# Feature Engineering Guide - Enhanced Land Property Value Prediction

## 🎯 Overview

This guide explains the comprehensive feature engineering implemented to improve land property value prediction accuracy. The enhanced model now includes **20+ engineered features** instead of the original 6.

---

## 📊 Feature Categories

### 1. **Size Features** (Property Dimensions)

| Feature | Description | Impact |
|---------|-------------|--------|
| `lot_area` | Total lot size in square meters | High - Directly affects value |
| `project_area` | Building/development area in sqm | High - Development density |
| `area_ratio` | `project_area / lot_area` (0-1) | Medium - Building density indicator |
| `size_difference` | `lot_area - project_area` | Medium - Open space available |
| `efficiency_ratio` | Building efficiency | Medium - Utilization rate |
| `lot_size_small` | Binary: < 100 sqm | Medium - Size category |
| `lot_size_medium` | Binary: 100-500 sqm | Medium - Size category |
| `lot_size_large` | Binary: ≥ 500 sqm | Medium - Size category |

**Why Important:**
- Larger lots typically have different per-sqm pricing
- Building density affects value
- Size categories capture non-linear relationships

---

### 2. **Location Features** (Geographic Context)

| Feature | Description | Impact |
|---------|-------------|--------|
| `project_location` | Location name (text) | High - Location premium |
| `location_category` | Encoded: Premium(3), Standard(2), Economy(1) | High - Location tier |
| `distance_to_center` | Distance to city center in km | High - Urban proximity |
| `latitude` | Geographic latitude | Medium - If available |
| `longitude` | Geographic longitude | Medium - If available |

**Why Important:**
- Location is the #1 factor in real estate value
- Distance to city center affects accessibility
- Location categories capture market segments

**Location Mapping:**
```python
Premium (3): Downtown, Urban Core, Commercial District
Standard (2): Suburban, Residential Area, Mixed Zone
Economy (1): Industrial Zone, Rural, Agricultural
```

---

### 3. **Temporal Features** (Time-Based Patterns)

| Feature | Description | Impact |
|---------|-------------|--------|
| `year` | Year of application | High - Inflation, market trends |
| `month` | Month (1-12) | Medium - Seasonal variations |
| `day_of_week` | Day of week (0-6) | Low - Application timing |
| `is_weekend` | Binary: Weekend application | Low - Timing indicator |
| `quarter` | Quarter (1-4) | Medium - Quarterly trends |
| `month_sin` | Cyclical: sin(2π × month/12) | Medium - Seasonality |
| `month_cos` | Cyclical: cos(2π × month/12) | Medium - Seasonality |

**Why Important:**
- Real estate markets have seasonal patterns
- Year captures inflation and market appreciation
- Cyclical encoding helps models understand seasonality

**Seasonal Patterns:**
- Spring (Mar-May): Higher activity
- Summer (Jun-Aug): Peak season
- Fall (Sep-Nov): Moderate
- Winter (Dec-Feb): Lower activity

---

### 4. **Zoning/Type Features** (Land Use Classification)

| Feature | Description | Impact |
|---------|-------------|--------|
| `project_type` | Type: residential, commercial, etc. | High - Use type |
| `project_type_encoded` | Numeric encoding | High - Model input |
| `site_zoning` | Zoning classification | High - Legal use |
| `zoning_category` | Encoded: Premium(3), Standard(2), Economy(1) | High - Zoning tier |
| `project_nature` | Nature: new_construction, renovation, etc. | Medium - Project type |
| `land_uses` | Multiple land uses | Low - Use diversity |

**Why Important:**
- Zoning determines allowed uses
- Commercial zones typically more valuable
- Project type affects development costs

**Zoning Mapping:**
```python
Premium (3): Commercial, Mixed-Use, Residential-High
Standard (2): Residential, Institutional
Economy (1): Agricultural, Industrial, Rural
```

---

### 5. **Interaction Features** (Combined Effects)

| Feature | Description | Impact |
|---------|-------------|--------|
| `size_location_interaction` | `lot_area × location_category` | High - Size premium by location |
| `type_size_interaction` | `project_type × lot_area` | Medium - Type-specific size effects |
| `year_location_interaction` | `(year - base) × location_category` | Medium - Location appreciation over time |

**Why Important:**
- Features interact in complex ways
- Large lots in premium locations = very high value
- Commercial projects on large lots = premium pricing
- Premium locations appreciate faster over time

**Example:**
- Small lot (100 sqm) × Premium location (3) = 300
- Large lot (500 sqm) × Premium location (3) = 1500
- Model learns that large premium lots are exponentially more valuable

---

### 6. **Market Features** (Economic Context)

| Feature | Description | Status |
|---------|-------------|--------|
| `inflation_factor` | Estimated inflation adjustment | Basic implementation |
| `years_since_base` | Years since base year | Basic implementation |
| `market_trend` | Market trend indicator | **TODO: Add external data** |
| `interest_rate` | Interest rates | **TODO: Add external data** |
| `GDP_growth` | Economic growth | **TODO: Add external data** |

**Why Important:**
- Market conditions affect property values
- Interest rates influence demand
- Economic indicators predict trends

**Future Enhancement:**
- Connect to economic data APIs
- Add inflation data from government sources
- Include interest rate data

---

## 🔧 Implementation Details

### Feature Engineering Function

The `_engineer_features()` method in `land_predictions.py` automatically:
1. Calculates size ratios and categories
2. Computes distance to city center (if coordinates available)
3. Encodes location and zoning categories
4. Creates temporal features with cyclical encoding
5. Generates interaction features
6. Handles missing values appropriately

### Feature Selection

The model automatically selects available features:
- Uses all numeric features
- Encodes categorical features
- Handles missing features gracefully

---

## 📈 Expected Improvements

### Before (6 Features):
- Features: lot_area, project_area, year, month, age, project_type
- R² Score: 0.15-0.20
- RMSE: 15,000-17,000 PHP/sqm

### After (20+ Features):
- Features: All above + engineered features
- R² Score: **0.50-0.70** (expected) ✅
- RMSE: **8,000-12,000 PHP/sqm** (expected) ✅
- **2-3x better accuracy**

---

## 🗺️ Location Data Requirements

### Current Implementation:
- Uses `project_location` (text field)
- Maps to location categories
- Calculates distance if `latitude`/`longitude` available

### Recommended Enhancements:

1. **Collect Coordinates:**
   ```sql
   -- Add latitude/longitude to all records
   UPDATE application_forms 
   SET latitude = 14.XXXX, longitude = 120.XXXX
   WHERE latitude IS NULL;
   ```

2. **Set City Center:**
   ```python
   # In land_predictions.py, update:
   city_center_lat = 14.5995  # Your city center
   city_center_lon = 120.9842
   ```

3. **Add Amenity Data:**
   - Create `amenities` table with locations
   - Calculate proximity to schools, hospitals, malls
   - Add as features

---

## 🏫 Amenity Proximity Features (Future)

### Recommended Amenity Data:

| Amenity Type | Why Important | Data Source |
|--------------|---------------|-------------|
| **Schools** | Education quality affects value | Department of Education |
| **Hospitals** | Healthcare access | Department of Health |
| **Shopping Malls** | Commercial activity | Business directories |
| **Parks** | Quality of life | City planning department |
| **Public Transport** | Accessibility | Transportation authority |
| **Highways** | Road access | DPWH |

### Implementation:
```python
# Calculate distance to nearest school
df['distance_to_school'] = calculate_distance(
    df['latitude'], df['longitude'],
    school_lat, school_lon
)

# Count schools within 5km
df['schools_within_5km'] = count_amenities_within_radius(
    df['latitude'], df['longitude'],
    school_locations, radius_km=5
)
```

---

## 📊 Market Condition Features (Future)

### Recommended Economic Data:

1. **Inflation Rate:**
   - Monthly CPI data
   - Source: PSA (Philippine Statistics Authority)

2. **Interest Rates:**
   - BSP policy rates
   - Source: Bangko Sentral ng Pilipinas

3. **GDP Growth:**
   - Quarterly GDP data
   - Source: PSA

4. **Unemployment Rate:**
   - Monthly unemployment
   - Source: PSA

### Implementation:
```python
# Load economic data
economic_data = load_economic_indicators()

# Merge with property data
df = df.merge(economic_data, on='year_month', how='left')

# Add features
df['inflation_rate'] = economic_data['cpi_change']
df['interest_rate'] = economic_data['bsp_rate']
df['gdp_growth'] = economic_data['gdp_growth']
```

---

## 🔍 Feature Importance Analysis

After training, check which features matter most:

```python
# Random Forest provides feature importance
importance = lp.feature_importance['land_cost']

# Sort by importance
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

for feature, score in sorted_features:
    print(f"{feature}: {score:.4f}")
```

**Expected Top Features:**
1. `location_category` - Location is #1 factor
2. `lot_area` - Size matters
3. `distance_to_center` - Urban proximity
4. `project_type_encoded` - Use type
5. `year` - Market trends
6. `size_location_interaction` - Combined effects

---

## ✅ Data Collection Checklist

### Immediate (Available Now):
- ✅ Lot area, project area
- ✅ Project type, location
- ✅ Year, month
- ✅ Age (from clients table)

### Short-term (Can Collect):
- ⚠️ Latitude, longitude (add to database)
- ⚠️ Site zoning (already in database)
- ⚠️ Location type (already in database)

### Long-term (External Data):
- ❌ Amenity locations (schools, hospitals, malls)
- ❌ Economic indicators (inflation, interest rates)
- ❌ Infrastructure data (roads, utilities)

---

## 🚀 Usage

### Automatic Feature Engineering:
```python
# Feature engineering happens automatically during training
lp = LandPredictions(db_config)
results = lp.train_all_models()  # Uses all engineered features
```

### Manual Feature Engineering:
```python
# Load data
df = lp.load_land_data()

# Engineer features
df = lp._engineer_features(df)

# Check available features
print(df.columns.tolist())
```

---

## 📝 Feature Engineering Best Practices

### 1. **Handle Missing Values:**
- Use median for numeric features
- Use mode for categorical features
- Create "missing" category for categorical

### 2. **Normalize Features:**
- Use StandardScaler for numeric features
- Use OneHotEncoder for categorical (if needed)
- Tree-based models handle raw features well

### 3. **Avoid Overfitting:**
- Don't create too many features (curse of dimensionality)
- Use feature importance to select best features
- Cross-validate model performance

### 4. **Domain Knowledge:**
- Include features that make business sense
- Real estate experts know what matters
- Test feature combinations

---

## 🎯 Next Steps

1. **Retrain Models:**
   ```bash
   POST /api/train
   ```
   Models will automatically use all engineered features

2. **Check Performance:**
   - Compare R² score (should be 0.50+)
   - Review feature importance
   - Test predictions

3. **Collect Additional Data:**
   - Add latitude/longitude to all records
   - Collect amenity locations
   - Gather economic indicators

4. **Monitor and Improve:**
   - Track prediction accuracy
   - Add new features as data becomes available
   - Retrain quarterly with new data

---

## 📚 References

- **Feature Engineering:** Scikit-learn documentation
- **Real Estate Valuation:** Industry best practices
- **Geographic Features:** Haversine formula for distances
- **Temporal Features:** Time series feature engineering

---

**Last Updated:** 2025  
**Feature Count:** 20+ engineered features  
**Expected Improvement:** 2-3x better accuracy

