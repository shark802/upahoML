# Fix Heroku Database Host - Quick Fix

## Problem
```
Cannot connect to 'localhost' from Heroku
Current DB_HOST: localhost
```

## Solution: Update DB_HOST on Heroku

### Step 1: Set the Correct Database Host

Run this command:

```bash
heroku config:set DB_HOST=srv1322.hstgr.io
```

### Step 2: Verify It's Set

```bash
heroku config:get DB_HOST
```

Should show: `srv1322.hstgr.io` (NOT localhost)

### Step 3: Verify All Database Config

```bash
heroku config | grep DB_
```

Should show:
```
DB_HOST: srv1322.hstgr.io
DB_USER: u520834156_uPAHOZone25
DB_PASSWORD: Y+;a+*1y
DB_NAME: u520834156_dbUPAHOZoning
DB_PORT: 3306
```

### Step 4: Test Connection

```bash
# Via command line
curl https://your-app-name.herokuapp.com/api/check

# Or visit in browser
https://your-app-name.herokuapp.com/api/check
```

## Complete Setup (All at Once)

If you need to set all database variables:

```bash
heroku config:set DB_HOST=srv1322.hstgr.io DB_USER=u520834156_uPAHOZone25 DB_PASSWORD=Y+;a+*1y DB_NAME=u520834156_dbUPAHOZoning DB_PORT=3306
```

## Alternative: Use IP Address

If hostname doesn't work, try IP:

```bash
heroku config:set DB_HOST=153.92.15.8
```

## After Setting DB_HOST

1. **Restart the app** (to pick up new config):
   ```bash
   heroku restart
   ```

2. **Check logs**:
   ```bash
   heroku logs --tail
   ```

3. **Test connection**:
   ```bash
   curl https://your-app-name.herokuapp.com/api/check
   ```

## Verify Current Config

```bash
# See all config
heroku config

# See just DB_HOST
heroku config:get DB_HOST
```

If it still shows `localhost`, the command didn't work. Make sure you're in the right directory and Heroku app is connected.

## Quick Check

```bash
# 1. Check current value
heroku config:get DB_HOST

# 2. If it shows "localhost", update it:
heroku config:set DB_HOST=srv1322.hstgr.io

# 3. Verify
heroku config:get DB_HOST

# 4. Restart app
heroku restart

# 5. Test
curl https://your-app-name.herokuapp.com/api/check
```



