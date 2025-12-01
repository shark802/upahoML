# Quick Deployment Commands

## If you already have Heroku app connected to Git:

```bash
# 1. Navigate to project
cd c:\xampp-25\htdocs\upaho_ml\upahoML

# 2. Commit changes
git add requirements.txt .slugignore app.py Procfile runtime.txt
git commit -m "Optimize for Heroku deployment"

# 3. Set database config (if not already set)
heroku config:set DB_HOST=localhost
heroku config:set DB_USER=u520834156_uPAHOZone25
heroku config:set DB_PASSWORD=Y+;a+*1y
heroku config:set DB_NAME=u520834156_dbUPAHOZoning

# 4. Deploy
git push heroku main

# 5. Check logs
heroku logs --tail
```

## If you need to create new Heroku app:

```bash
# 1. Login
heroku login

# 2. Create app
heroku create your-app-name

# 3. Set database config
heroku config:set DB_HOST=localhost
heroku config:set DB_USER=u520834156_uPAHOZone25
heroku config:set DB_PASSWORD=Y+;a+*1y
heroku config:set DB_NAME=u520834156_dbUPAHOZoning

# 4. Commit and deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# 5. Open app
heroku open
```

## Expected Slug Size

With current minimal `requirements.txt`:
- **Estimated size**: ~200-250MB ✅ (well under 500MB limit)

## What Was Removed

- ❌ TensorFlow (~500MB)
- ❌ Prophet (~200MB)  
- ❌ XGBoost (~100MB)
- ❌ pmdarima (~50MB)

**Total saved**: ~850MB

## What Still Works

- ✅ Land cost prediction (Linear Regression)
- ✅ Approval prediction (Random Forest/Gradient Boosting)
- ✅ Processing time prediction (Random Forest/Gradient Boosting)
- ✅ Time series (ARIMA - manual parameters)
- ✅ All core functionality

## If Still Too Large

Remove statsmodels (loses ARIMA time series):
```bash
# Edit requirements.txt - comment out:
# statsmodels>=0.14.0
```

This brings size down to ~150MB.



