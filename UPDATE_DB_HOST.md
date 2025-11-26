# Update Database Host

## Set Database Host on Heroku

Run these commands to update your database host:

```bash
# Set the database host
heroku config:set DB_HOST=srv1322.hstgr.io

# Verify it's set correctly
heroku config:get DB_HOST

# Should output: srv1322.hstgr.io
```

## Alternative: Use IP Address

If the hostname doesn't work, you can use the IP address:

```bash
heroku config:set DB_HOST=153.92.15.8
```

## Verify All Database Config

```bash
# Check all database settings
heroku config | grep DB_

# Should show:
# DB_HOST: srv1322.hstgr.io
# DB_USER: u520834156_uPAHOZone25
# DB_PASSWORD: Y+;a+*1y
# DB_NAME: u520834156_dbUPAHOZoning
```

## Test Connection

After updating, test the connection:

```bash
# Test via API
curl https://your-app-name.herokuapp.com/api/check
```

Or visit in browser:
```
https://your-app-name.herokuapp.com/api/check
```

Should now show:
```json
{
  "database": {
    "connected": true,
    "records_available": 1234,
    "host": "srv1322.hstgr.io"
  }
}
```

## If Connection Still Fails

1. **Check database allows remote connections:**
   - Log into your hosting control panel
   - Enable "Remote MySQL" or "Allow Remote Connections"
   - Add Heroku IPs to allowed hosts (or allow all for testing)

2. **Check firewall:**
   - Ensure port 3306 is open
   - Allow connections from Heroku IPs

3. **Verify credentials:**
   ```bash
   heroku config
   ```
   Make sure DB_USER, DB_PASSWORD, and DB_NAME are correct.

## Next Steps After Connection Works

1. **Train models:**
   ```bash
   curl https://your-app-name.herokuapp.com/api/train
   ```

2. **Test predictions:**
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

