# Heroku Slug Size Optimization Guide

## Current Issue
Slug size: **1.2GB** (exceeds 500MB limit)

## Solutions Applied

### 1. ✅ Removed TensorFlow
- **Size saved**: ~500MB
- **Impact**: LSTM models won't work, but code handles this gracefully
- **Status**: Already removed from `requirements.txt`

### 2. Additional Optimizations

#### Option A: Remove Prophet (if not using time series)
If you're not using Prophet for time series forecasting:
```bash
# Remove from requirements.txt
# prophet>=1.1.0
```
**Size saved**: ~100-200MB

#### Option B: Remove XGBoost (if not using)
If you're not using XGBoost models:
```bash
# Remove from requirements.txt
# xgboost>=2.0.0
```
**Size saved**: ~50-100MB

#### Option C: Use Minimal Requirements
Use `requirements-minimal.txt` instead:
```bash
cp requirements-minimal.txt requirements.txt
```
This removes XGBoost, Prophet, and pmdarima.

## Recommended Approach

### Step 1: Remove TensorFlow (DONE ✅)
Already removed from `requirements.txt`

### Step 2: Test Current Size
```bash
git add requirements.txt
git commit -m "Remove TensorFlow to reduce slug size"
git push heroku main
```

### Step 3: If Still Too Large
Remove Prophet if not needed:
```bash
# Edit requirements.txt and comment out:
# prophet>=1.1.0
```

### Step 4: If Still Too Large
Use minimal requirements:
```bash
cp requirements-minimal.txt requirements.txt
git add requirements.txt
git commit -m "Use minimal requirements"
git push heroku main
```

## Alternative: External Model Storage

If you need all dependencies but models are large:

1. **Store models in S3/Cloud Storage**
2. **Download on app startup** (add to `app.py`)
3. **Exclude models from Git** (already in `.slugignore`)

Example code to add to `app.py`:
```python
import boto3
import os

def download_models_from_s3():
    """Download models from S3 on startup"""
    s3 = boto3.client('s3')
    bucket = os.environ.get('S3_BUCKET')
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Download model files
    model_files = ['land_cost_model.pkl', 'approval_model.pkl', ...]
    for file in model_files:
        s3.download_file(bucket, f'models/{file}', 
                        os.path.join(models_dir, file))
```

## Check Slug Size

After deployment, check size:
```bash
heroku run du -sh /app
```

## Expected Sizes

| Configuration | Approximate Size |
|--------------|------------------|
| Full (with TensorFlow) | ~1.2GB ❌ |
| Without TensorFlow | ~700MB ⚠️ |
| Without TensorFlow + Prophet | ~500MB ✅ |
| Minimal (core only) | ~300MB ✅ |

## Current Status

✅ TensorFlow removed
⚠️ Still may need to remove Prophet or XGBoost
✅ `.slugignore` configured to exclude models

## Next Steps

1. Try deployment with current `requirements.txt` (TensorFlow removed)
2. If still fails, remove Prophet
3. If still fails, use `requirements-minimal.txt`



