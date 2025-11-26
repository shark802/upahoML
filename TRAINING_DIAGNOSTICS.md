# Training Diagnostics Guide

## Issue: Empty Results from Training

If you get:
```json
{
  "success": true,
  "message": "Models trained successfully",
  "results": {}
}
```

This means training ran but returned no results. Possible causes:

## Check Database Data

### 1. Check if Database Has Data

```bash
# Via API
curl https://your-app-name.herokuapp.com/api/check

# Should show:
# "records_available": 1234
```

### 2. Check Data Quality

The training needs records with:
- ✅ `project_cost_numeric` (NOT NULL, > 0)
- ✅ `lot_area` (NOT NULL, > 0)
- ✅ `project_area` (NOT NULL)
- ✅ `project_type` (NOT NULL)
- ✅ `created_at` (NOT NULL)

### 3. Check Heroku Logs

```bash
heroku logs --tail
```

Look for:
- "Loaded X records"
- "Training samples: X"
- Any error messages

## Common Issues

### Issue 1: No Data in Database

**Symptoms:**
- Empty results
- "No land cost data available" in logs

**Solution:**
- Ensure database has records
- Check records have cost data filled
- Minimum 50-100 records recommended

### Issue 2: All Data Filtered Out

**Symptoms:**
- "No valid data after preprocessing" in logs
- Records exist but training fails

**Solution:**
- Check for extreme outliers in cost data
- Verify `project_cost_numeric / lot_area` is reasonable
- Check for missing required fields

### Issue 3: Training Returns Empty Dict

**Symptoms:**
- Training completes but results is empty
- No error message

**Solution:**
- Check logs for actual error
- Verify database connection
- Check data quality

## Debug Steps

### Step 1: Check Database Connection

```bash
curl https://your-app-name.herokuapp.com/api/check
```

### Step 2: Check Database Records

Run this query in your database:
```sql
SELECT COUNT(*) as total,
       SUM(CASE WHEN project_cost_numeric IS NOT NULL AND project_cost_numeric > 0 THEN 1 ELSE 0 END) as with_cost,
       SUM(CASE WHEN lot_area IS NOT NULL AND lot_area > 0 THEN 1 ELSE 0 END) as with_area,
       SUM(CASE WHEN project_cost_numeric IS NOT NULL AND lot_area IS NOT NULL AND lot_area > 0 THEN 1 ELSE 0 END) as valid_records
FROM application_forms;
```

### Step 3: Check Training Logs

```bash
heroku logs --tail | grep -i "training\|error\|loaded"
```

### Step 4: Test Training Again

```bash
curl https://your-app-name.herokuapp.com/api/train
```

## Expected Successful Response

```json
{
  "success": true,
  "message": "Models trained successfully",
  "results": {
    "land_cost": {
      "mae": 123.45,
      "mse": 23456.78,
      "rmse": 234.56,
      "r2_score": 0.85,
      "features": ["lot_area", "project_area", "year", "month", "age", "project_type_encoded"]
    },
    "models_saved": true
  },
  "models_loaded": true,
  "model_files": 3
}
```

## Next Steps

1. ✅ Check database has data
2. ✅ Verify data quality
3. ✅ Check Heroku logs
4. ✅ Retry training
5. ✅ Test prediction after training

