# Insert Training Data Guide

## Quick Insert via API

### Option 1: Insert 500 Records (Default)

```bash
# Via command line
curl https://your-app-name.herokuapp.com/api/insert_training_data

# Or visit in browser
https://your-app-name.herokuapp.com/api/insert_training_data
```

### Option 2: Insert Custom Number of Records

```bash
# Insert 1000 records
curl "https://your-app-name.herokuapp.com/api/insert_training_data?num_records=1000"

# Insert 2000 records
curl "https://your-app-name.herokuapp.com/api/insert_training_data?num_records=2000"
```

**Note**: Maximum 5000 records per request

## What Data Gets Inserted

The script generates realistic training data with:

- ✅ **Project Types**: residential, commercial, industrial, agricultural, etc.
- ✅ **Cost Data**: Realistic `project_cost_numeric` based on project type and location
- ✅ **Area Data**: `lot_area` and `project_area` with realistic ratios
- ✅ **Location Data**: Various locations (Downtown, Barangays, etc.)
- ✅ **Time Data**: Records spanning last 3 years
- ✅ **Client Data**: Age, gender for each application

## Complete Workflow

### Step 1: Insert Training Data

```bash
# Insert 1000 records
curl "https://your-app-name.herokuapp.com/api/insert_training_data?num_records=1000"
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Successfully inserted 1000 training records",
  "records_inserted": 1000,
  "total_records_in_db": 1500
}
```

### Step 2: Verify Data

```bash
curl https://your-app-name.herokuapp.com/api/check
```

Should show `records_available` > 0

### Step 3: Train Models

```bash
curl https://your-app-name.herokuapp.com/api/train
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Models trained successfully",
  "results": {
    "land_cost": {
      "mae": 123.45,
      "rmse": 234.56,
      "r2_score": 0.85
    },
    "models_saved": true
  }
}
```

### Step 4: Test Prediction

```bash
curl -X POST https://your-app-name.herokuapp.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_type": "land_cost_future",
    "target_years": 10,
    "data": {
      "lot_area": 200,
      "project_area": 150,
      "project_type": "residential",
      "location": "Downtown"
    }
  }'
```

## Data Requirements

For training to work, you need:
- **Minimum**: 50-100 records
- **Recommended**: 500-1000 records
- **Optimal**: 1000+ records

## Data Characteristics

The generated data includes:
- **Cost per sqm**: 3,000 - 60,000 PHP (varies by project type)
- **Lot areas**: 80 - 10,000 sqm (varies by project type)
- **Time span**: Last 3 years
- **Locations**: Multiple locations with different pricing
- **Project types**: 8 different types with realistic distribution

## Troubleshooting

### Issue: "Failed to insert data"

**Check:**
1. Database connection works
2. Tables `clients` and `application_forms` exist
3. User has INSERT permissions

### Issue: "Duplicate entry"

**Solution**: The script handles duplicates automatically. If you get this error, some records may have been inserted.

### Issue: "Table doesn't exist"

**Solution**: Ensure database tables are created. Check your database schema.

## Manual Insert (Alternative)

If API doesn't work, you can run the script locally:

```bash
# Set environment variables
export DB_HOST=srv1322.hstgr.io
export DB_USER=u520834156_uPAHOZone25
export DB_PASSWORD=Y+;a+*1y
export DB_NAME=u520834156_dbUPAHOZoning
export DB_PORT=3306

# Run script
python generate_mock_data.py
```

## Verify Data Quality

After inserting, check:

```sql
-- Count records with cost data
SELECT COUNT(*) 
FROM application_forms 
WHERE project_cost_numeric IS NOT NULL 
  AND project_cost_numeric > 0
  AND lot_area IS NOT NULL 
  AND lot_area > 0;

-- Check cost per sqm distribution
SELECT 
    project_type,
    COUNT(*) as count,
    AVG(project_cost_numeric / lot_area) as avg_cost_per_sqm,
    MIN(project_cost_numeric / lot_area) as min_cost,
    MAX(project_cost_numeric / lot_area) as max_cost
FROM application_forms
WHERE project_cost_numeric IS NOT NULL 
  AND lot_area IS NOT NULL 
  AND lot_area > 0
GROUP BY project_type;
```

## Next Steps

1. ✅ Insert training data (500-1000 records recommended)
2. ✅ Verify data inserted: `/api/check`
3. ✅ Train models: `/api/train`
4. ✅ Test predictions
5. ✅ Use in production



