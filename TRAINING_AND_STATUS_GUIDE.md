# Training Models and Status Check Guide

## New Features Added

✅ **Auto-training**: Models automatically train if not found (on first prediction request)
✅ **Database connection check**: Verifies database connection before operations
✅ **Training endpoint**: `/api/train` - Manually train models
✅ **Status check endpoint**: `/api/check` - Check database and model status

## Endpoints

### 1. Check Status
**GET** `/api/check`

Check database connection and model status.

**Response:**
```json
{
  "success": true,
  "database": {
    "connected": true,
    "records_available": 1234,
    "host": "localhost",
    "database": "u520834156_dbUPAHOZoning"
  },
  "models": {
    "loaded": true,
    "models_dir": "/app/models",
    "files": ["land_cost_model.pkl", "project_type_encoder.pkl", ...],
    "model_count": 3
  }
}
```

### 2. Train Models
**GET/POST** `/api/train`

Train models from database data.

**Response:**
```json
{
  "success": true,
  "message": "Models trained successfully",
  "results": {
    "land_cost": {
      "mae": 123.45,
      "rmse": 234.56,
      "r2_score": 0.85
    }
  }
}
```

## How It Works

### Auto-Training
When a prediction is requested:
1. ✅ Check database connection
2. ✅ Try to load existing models
3. ✅ If no models found, automatically train from database
4. ✅ Reload models after training
5. ✅ Make prediction

### Manual Training
You can also train models manually:

**Via Browser:**
```
https://your-app-name.herokuapp.com/api/train
```

**Via cURL:**
```bash
curl https://your-app-name.herokuapp.com/api/train
```

**Via PHP:**
```php
$ch = curl_init('https://your-app-name.herokuapp.com/api/train');
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
$result = json_decode($response, true);
```

## Troubleshooting

### Issue: "No trained models available"

**Solution 1: Check database connection**
```bash
curl https://your-app-name.herokuapp.com/api/check
```

**Solution 2: Train models manually**
```bash
curl https://your-app-name.herokuapp.com/api/train
```

**Solution 3: Check database has data**
- Ensure `application_forms` table has records
- Ensure records have `project_cost_numeric`, `lot_area`, `project_area` filled
- Minimum 50-100 records recommended for training

### Issue: "Database connection failed"

**Check:**
1. Environment variables are set correctly:
   ```bash
   heroku config
   ```

2. Database allows connections from Heroku IPs

3. Database credentials are correct:
   ```bash
   heroku config:set DB_HOST=your-host
   heroku config:set DB_USER=your-user
   heroku config:set DB_PASSWORD=your-password
   heroku config:set DB_NAME=your-database
   ```

### Issue: "No valid data after preprocessing"

**Causes:**
- Database has no records with cost data
- Records missing required fields (`project_cost_numeric`, `lot_area`, etc.)
- All records filtered out as outliers

**Solution:**
- Ensure database has sufficient data (100+ records recommended)
- Check data quality in database
- Verify required fields are populated

## Testing

### 1. Check Status
```bash
curl https://your-app-name.herokuapp.com/api/check
```

### 2. Train Models
```bash
curl https://your-app-name.herokuapp.com/api/train
```

### 3. Make Prediction
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

## Database Requirements

For training to work, your database needs:

1. **Table**: `application_forms`
2. **Required Fields**:
   - `project_type` (NOT NULL)
   - `project_cost_numeric` (NOT NULL, > 0)
   - `lot_area` (NOT NULL, > 0)
   - `project_area` (NOT NULL)
   - `created_at` (NOT NULL)
   - `project_location` (optional but recommended)

3. **Minimum Data**:
   - At least 50-100 records for basic training
   - More records = better model accuracy
   - Records should span multiple years for time-based features

4. **Data Quality**:
   - `project_cost_numeric / lot_area` should be reasonable (not extreme outliers)
   - Missing values will be filled with medians

## Logs

Check Heroku logs to see training progress:
```bash
heroku logs --tail
```

You should see:
- "Database connection successful"
- "No trained models found. Attempting to train models..."
- "Models trained successfully"
- "Loaded land_cost model"

## Next Steps

1. ✅ Deploy updated code
2. ✅ Check status: `GET /api/check`
3. ✅ Train models: `GET /api/train` (or let auto-training handle it)
4. ✅ Test predictions
5. ✅ Monitor logs for any issues



