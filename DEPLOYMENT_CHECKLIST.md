# Heroku Deployment Checklist

## ✅ Files Created for Heroku Deployment

1. **requirements.txt** - Python dependencies
2. **runtime.txt** - Python version specification
3. **Procfile** - Defines web process
4. **app.py** - Flask API server
5. **.slugignore** - Files to exclude from deployment
6. **.gitignore** - Git ignore patterns

## Pre-Deployment Steps

### 1. Verify Files Exist
```bash
ls -la requirements.txt runtime.txt Procfile app.py
```

### 2. Check Python Version
- Current: Python 3.11.9 (in runtime.txt)
- Verify compatibility with your code

### 3. Review Dependencies
- Check `requirements.txt` for all needed packages
- Note: TensorFlow is large (~500MB) - remove if not using LSTM

### 4. Database Configuration
- Set up cloud database (Heroku Postgres, AWS RDS, etc.)
- Get connection credentials
- Will be set as environment variables

### 5. Model Files
- Train models locally first
- Models should be in `models/` directory
- Or plan to download from external storage (S3, etc.)

## Deployment Commands

```bash
# 1. Login
heroku login

# 2. Create app (if new)
heroku create your-app-name

# 3. Set database config
heroku config:set DB_HOST=your-host
heroku config:set DB_USER=your-user
heroku config:set DB_PASSWORD=your-password
heroku config:set DB_NAME=your-database

# 4. Deploy
git add .
git commit -m "Prepare for Heroku deployment"
git push heroku main

# 5. Check logs
heroku logs --tail
```

## Post-Deployment

1. **Test API**: Visit `https://your-app-name.herokuapp.com/`
2. **Check Health**: Should return JSON with status
3. **Test Endpoints**: Use curl or Postman to test prediction endpoints
4. **Monitor Logs**: `heroku logs --tail`

## Common Issues

### Issue: Build fails - "No module named X"
**Solution**: Add missing package to `requirements.txt`

### Issue: Models not found
**Solution**: 
- Train models and commit to Git (if small)
- Or download from S3 on startup
- Or use Heroku's ephemeral filesystem

### Issue: Database connection fails
**Solution**: 
- Verify environment variables are set
- Check database allows Heroku IPs
- Test connection from local machine

### Issue: Slug size too large (>500MB)
**Solution**:
- Remove TensorFlow if not needed
- Store models externally (S3)
- Use .slugignore to exclude large files

## Optional Optimizations

1. **Remove TensorFlow** (if not using LSTM):
   - Remove from `requirements.txt`
   - Saves ~500MB

2. **Use External Model Storage**:
   - Upload models to S3
   - Download on app startup
   - Add download code to `app.py`

3. **Add Caching**:
   - Use Redis for model caching
   - Reduces database load

4. **Add Monitoring**:
   - Heroku Metrics
   - Log aggregation
   - Error tracking (Sentry)

