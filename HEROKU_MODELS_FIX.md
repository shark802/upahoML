# Heroku Models Directory Fix

## Problem
Error: `errno 30 read only file system: /app/../models`

This happens because:
1. Heroku's filesystem is read-only except for `/tmp` and the app directory
2. The code was trying to create `../models` (outside the app directory)
3. That location is read-only on Heroku

## Solution Applied

✅ Changed models directory from `../models` to `./models` (within app directory)
✅ Added fallback to `/tmp/models` if app directory is read-only
✅ Updated both `land_predictions.py` and `ml_predictive_analytics.py`

## Models Directory Location

**Before (doesn't work on Heroku):**
```python
self.models_dir = os.path.join(self.base_path, '..', 'models')  # /app/../models
```

**After (works on Heroku):**
```python
self.models_dir = os.path.join(self.base_path, 'models')  # /app/models
```

## Important Notes

### 1. Models Directory in Git
The `models/` directory should now be included in your Git repository if you have pre-trained models:

```bash
# Add models directory to Git (if you have trained models)
git add models/
git commit -m "Add trained models"
git push heroku main
```

### 2. Training Models on Heroku
If you need to train models on Heroku, you can:

**Option A: Train locally and commit models**
```bash
# Train locally
python -c "from land_predictions import LandPredictions; import json; db_config = {...}; lp = LandPredictions(db_config); lp.train_all_models()"

# Commit models
git add models/
git commit -m "Add trained models"
git push heroku main
```

**Option B: Train on Heroku (one-time)**
```bash
heroku run python -c "from land_predictions import LandPredictions; import json; db_config = {...}; lp = LandPredictions(db_config); lp.train_all_models()"
```

**Note**: Models trained on Heroku will be lost when the dyno restarts (ephemeral filesystem). Use Option A for persistence.

### 3. Using External Storage (Recommended for Production)

For production, consider storing models in:
- **AWS S3** - Download on app startup
- **Heroku Postgres** - Store as binary
- **Git Repository** - Commit trained models

Example S3 download on startup:
```python
import boto3
import os

def download_models_from_s3():
    s3 = boto3.client('s3')
    bucket = os.environ.get('S3_BUCKET')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for file in ['land_cost_model.pkl', 'project_type_encoder.pkl', ...]:
        s3.download_file(bucket, f'models/{file}', os.path.join(models_dir, file))
```

## Deployment Steps

1. **Deploy the fix:**
   ```bash
   git add land_predictions.py ml_predictive_analytics.py
   git commit -m "Fix models directory path for Heroku"
   git push heroku main
   ```

2. **If you have trained models, add them:**
   ```bash
   git add models/
   git commit -m "Add trained models"
   git push heroku main
   ```

3. **Or train models on Heroku (one-time):**
   ```bash
   heroku run python -c "from land_predictions import LandPredictions; import json; import os; db_config = {'host': os.environ.get('DB_HOST'), 'user': os.environ.get('DB_USER'), 'password': os.environ.get('DB_PASSWORD'), 'database': os.environ.get('DB_NAME')}; lp = LandPredictions(db_config); lp.train_all_models()"
   ```

## Verify Fix

After deployment, check logs:
```bash
heroku logs --tail
```

You should no longer see the "read only file system" error.

## Directory Structure

```
/app/
├── app.py
├── land_predictions.py
├── ml_predictive_analytics.py
├── models/              ← Models directory (now in app root)
│   ├── land_cost_model.pkl
│   ├── project_type_encoder.pkl
│   └── ...
└── ...
```



