# Complete Database Connection Setup Guide

## Step-by-Step Instructions

### Step 1: Gather Your Database Information

You need the following information:
- ✅ **Database Host**: `srv1322.hstgr.io` (or `153.92.15.8`)
- ✅ **Database User**: `u520834156_uPAHOZone25`
- ✅ **Database Password**: `Y+;a+*1y`
- ✅ **Database Name**: `u520834156_dbUPAHOZoning`
- ✅ **Database Port**: `3306` (default MySQL port)

### Step 2: Login to Heroku CLI

```bash
heroku login
```

This will open a browser window. Click "Log in" to authenticate.

### Step 3: Navigate to Your Project (if not already there)

```bash
cd c:\xampp-25\htdocs\upaho_ml\upahoML
```

### Step 4: Check Current Heroku App

```bash
# Check if you have a Heroku app connected
git remote -v

# If you see "heroku" remote, you're good
# If not, create one:
heroku create your-app-name
```

### Step 5: Set All Database Environment Variables

Set each database configuration variable:

```bash
# Set database host
heroku config:set DB_HOST=srv1322.hstgr.io

# Set database user
heroku config:set DB_USER=u520834156_uPAHOZone25

# Set database password
heroku config:set DB_PASSWORD=Y+;a+*1y

# Set database name
heroku config:set DB_NAME=u520834156_dbUPAHOZoning

# Set database port (optional, 3306 is default)
heroku config:set DB_PORT=3306
```

**Or set all at once:**
```bash
heroku config:set DB_HOST=srv1322.hstgr.io DB_USER=u520834156_uPAHOZone25 DB_PASSWORD=Y+;a+*1y DB_NAME=u520834156_dbUPAHOZoning DB_PORT=3306
```

### Step 6: Verify Configuration

Check that all variables are set correctly:

```bash
# View all database-related config
heroku config | grep DB_

# Or view all config
heroku config
```

**Expected output:**
```
DB_HOST: srv1322.hstgr.io
DB_NAME: u520834156_dbUPAHOZoning
DB_PASSWORD: Y+;a+*1y
DB_PORT: 3306
DB_USER: u520834156_uPAHOZone25
```

### Step 7: Test Database Connection

#### Option A: Via API Endpoint (Recommended)

```bash
# Get your app URL first
heroku info

# Then test connection
curl https://your-app-name.herokuapp.com/api/check
```

**Or visit in browser:**
```
https://your-app-name.herokuapp.com/api/check
```

**Expected successful response:**
```json
{
  "success": true,
  "database": {
    "connected": true,
    "records_available": 1234,
    "host": "srv1322.hstgr.io",
    "port": 3306,
    "database": "u520834156_dbUPAHOZoning"
  },
  "models": {
    "loaded": false,
    "models_dir": "/app/models",
    "files": []
  }
}
```

#### Option B: Via Heroku CLI

```bash
heroku run python -c "
import mysql.connector
import os
try:
    conn = mysql.connector.connect(
        host=os.environ.get('DB_HOST'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD'),
        database=os.environ.get('DB_NAME'),
        port=int(os.environ.get('DB_PORT', 3306)),
        connect_timeout=10
    )
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM application_forms')
    count = cursor.fetchone()[0]
    print(f'✅ Connection successful!')
    print(f'✅ Records found: {count}')
    cursor.close()
    conn.close()
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
```

### Step 8: Enable Remote MySQL Access (If Connection Fails)

If you get connection errors, you need to enable remote MySQL access:

1. **Log into your hosting control panel** (cPanel, etc.)

2. **Find "Remote MySQL" or "MySQL Access Hosts"**

3. **Add Heroku IPs** (or allow all for testing):
   - Option A: Add specific Heroku IP ranges
   - Option B: Add `%` to allow all IPs (less secure, but works for testing)

4. **Save changes**

### Step 9: Train Models (After Connection Works)

Once database is connected, train the models:

```bash
# Via API
curl https://your-app-name.herokuapp.com/api/train

# Or visit in browser
https://your-app-name.herokuapp.com/api/train
```

**Expected response:**
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

### Step 10: Test Prediction

After models are trained, test a prediction:

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
      "location": "Downtown",
      "year": 2024,
      "month": 1,
      "age": 35
    }
  }'
```

## Troubleshooting

### Issue: "Error 2003 - Can't connect to MySQL server"

**Possible causes:**
1. Database host is incorrect
2. Remote MySQL access not enabled
3. Firewall blocking port 3306
4. Database server is down

**Solutions:**
```bash
# 1. Verify host is correct
heroku config:get DB_HOST
# Should show: srv1322.hstgr.io

# 2. Try IP address instead
heroku config:set DB_HOST=153.92.15.8

# 3. Check if hostname resolves
heroku run nslookup srv1322.hstgr.io

# 4. Enable remote MySQL in hosting panel
```

### Issue: "Access denied for user"

**Solutions:**
```bash
# 1. Verify username
heroku config:get DB_USER

# 2. Verify password (be careful with special characters)
heroku config:get DB_PASSWORD

# 3. Check if user has remote access permissions
```

### Issue: "Unknown database"

**Solutions:**
```bash
# 1. Verify database name
heroku config:get DB_NAME

# 2. Check database exists
# Log into your hosting panel and verify database name
```

### Issue: Connection timeout

**Solutions:**
1. Check database server is running
2. Verify firewall allows port 3306
3. Check network connectivity
4. Try IP address instead of hostname

## Complete Setup Checklist

- [ ] Heroku CLI installed and logged in
- [ ] Heroku app created
- [ ] DB_HOST set to `srv1322.hstgr.io`
- [ ] DB_USER set to `u520834156_uPAHOZone25`
- [ ] DB_PASSWORD set to `Y+;a+*1y`
- [ ] DB_NAME set to `u520834156_dbUPAHOZoning`
- [ ] DB_PORT set to `3306`
- [ ] All config variables verified
- [ ] Database connection tested successfully
- [ ] Remote MySQL access enabled (if needed)
- [ ] Models trained successfully
- [ ] Prediction tested and working

## Quick Reference Commands

```bash
# Set all config at once
heroku config:set DB_HOST=srv1322.hstgr.io DB_USER=u520834156_uPAHOZone25 DB_PASSWORD=Y+;a+*1y DB_NAME=u520834156_dbUPAHOZoning DB_PORT=3306

# View all config
heroku config

# Test connection
curl https://your-app-name.herokuapp.com/api/check

# Train models
curl https://your-app-name.herokuapp.com/api/train

# View logs
heroku logs --tail
```

## Next Steps After Setup

1. ✅ Database connected
2. ✅ Models trained
3. ✅ Test predictions
4. ✅ Update PHP to use API
5. ✅ Monitor logs for any issues

## Security Notes

⚠️ **Important:**
- Never commit database credentials to Git
- Use Heroku config vars for all sensitive data
- Consider using SSL connections for production
- Restrict database access to specific IPs when possible
- Rotate passwords regularly

## Need Help?

If connection still fails:
1. Check Heroku logs: `heroku logs --tail`
2. Verify database credentials in hosting panel
3. Test connection from local machine first
4. Contact hosting provider support for remote MySQL setup

