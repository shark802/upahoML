# Heroku CLI Deployment Guide - Step by Step

## Prerequisites Check

Before starting, ensure you have:
- ✅ Heroku account (sign up at https://heroku.com)
- ✅ Heroku CLI installed (download from https://devcenter.heroku.com/articles/heroku-cli)
- ✅ Git repository initialized and connected
- ✅ Your app code ready

## Step-by-Step Deployment

### Step 1: Verify Heroku CLI Installation

Open your terminal/command prompt and check:

```bash
heroku --version
```

If not installed, download from: https://devcenter.heroku.com/articles/heroku-cli

### Step 2: Login to Heroku

```bash
heroku login
```

This will open a browser window. Click "Log in" to authenticate.

### Step 3: Navigate to Your Project Directory

```bash
cd c:\xampp-25\htdocs\upaho_ml\upahoML
```

Or wherever your project is located.

### Step 4: Check Git Status

```bash
git status
```

Make sure all your files are committed:
```bash
git add .
git commit -m "Prepare for Heroku deployment with minimal requirements"
```

### Step 5: Create Heroku App

```bash
heroku create your-app-name
```

Replace `your-app-name` with your desired app name (must be unique). Example:
```bash
heroku create upaho-ml-predictions
```

**Note**: If you already have a Heroku app connected, skip this step. Check with:
```bash
git remote -v
```

If you see `heroku` remote, you're already connected.

### Step 6: Set Database Environment Variables

Set your database credentials as environment variables:

```bash
heroku config:set DB_HOST=localhost
heroku config:set DB_USER=u520834156_uPAHOZone25
heroku config:set DB_PASSWORD=Y+;a+*1y
heroku config:set DB_NAME=u520834156_dbUPAHOZoning
```

**Important**: If your database is not accessible from Heroku (localhost won't work), you need:
- A cloud database (AWS RDS, Heroku Postgres, etc.)
- Update the DB_HOST to your cloud database host

### Step 7: Verify Requirements File

Make sure `requirements.txt` is in the root directory and contains minimal dependencies:

```bash
cat requirements.txt
```

Or on Windows:
```bash
type requirements.txt
```

### Step 8: Deploy to Heroku

```bash
git push heroku main
```

Or if your default branch is `master`:
```bash
git push heroku master
```

**Watch the output carefully!** You'll see:
- Building dependencies
- Installing packages
- Compiling slug
- Starting web process

### Step 9: Check Deployment Status

After deployment, check logs:

```bash
heroku logs --tail
```

This shows real-time logs. Press `Ctrl+C` to exit.

### Step 10: Test Your API

Open your app in browser:
```bash
heroku open
```

Or visit: `https://your-app-name.herokuapp.com/`

You should see a JSON response with API endpoints.

## Troubleshooting

### Issue: "No app specified"

**Solution**: Make sure you're in the right directory and Heroku remote is set:
```bash
git remote add heroku https://git.heroku.com/your-app-name.git
```

### Issue: "Slug size too large"

**Solution**: The current `requirements.txt` should be minimal. If still too large:

1. Remove statsmodels if not using ARIMA:
   ```bash
   # Edit requirements.txt - comment out statsmodels
   ```

2. Check what's taking space:
   ```bash
   heroku run du -sh /app
   ```

### Issue: "Build failed - No module named X"

**Solution**: Add missing package to `requirements.txt`:
```bash
# Edit requirements.txt and add the missing package
git add requirements.txt
git commit -m "Add missing dependency"
git push heroku main
```

### Issue: "Database connection failed"

**Solution**: 
1. Verify environment variables:
   ```bash
   heroku config
   ```

2. Make sure your database allows connections from Heroku IPs
3. If using localhost, switch to cloud database

### Issue: "Models not found"

**Solution**: 
1. Train models locally first
2. Commit model files (if small):
   ```bash
   git add models/
   git commit -m "Add trained models"
   git push heroku main
   ```

3. Or use external storage (S3) and download on startup

## Useful Commands

### View App Info
```bash
heroku info
```

### View Environment Variables
```bash
heroku config
```

### Set Environment Variable
```bash
heroku config:set KEY=value
```

### Remove Environment Variable
```bash
heroku config:unset KEY
```

### View Logs
```bash
heroku logs --tail
```

### Open App in Browser
```bash
heroku open
```

### Run One-Off Command
```bash
heroku run python your_script.py
```

### Check App Status
```bash
heroku ps
```

### Restart App
```bash
heroku restart
```

## Post-Deployment Checklist

- [ ] App deploys successfully
- [ ] Health check endpoint works (`/`)
- [ ] Database connection works
- [ ] Models load correctly
- [ ] API endpoints respond
- [ ] Logs show no errors

## Next Steps After Deployment

1. **Test API Endpoints**:
   ```bash
   curl https://your-app-name.herokuapp.com/
   ```

2. **Monitor Logs**:
   ```bash
   heroku logs --tail
   ```

3. **Set Up Monitoring** (optional):
   - Heroku Metrics
   - Error tracking (Sentry)

4. **Scale if Needed**:
   ```bash
   heroku ps:scale web=1
   ```

## Quick Reference

```bash
# Full deployment workflow
cd your-project-directory
git add .
git commit -m "Deploy to Heroku"
heroku login
heroku create your-app-name
heroku config:set DB_HOST=your-host DB_USER=your-user DB_PASSWORD=your-pass DB_NAME=your-db
git push heroku main
heroku logs --tail
```

## Need Help?

- Heroku Docs: https://devcenter.heroku.com/articles/getting-started-with-python
- Heroku Support: https://help.heroku.com/
- Check logs: `heroku logs --tail`



