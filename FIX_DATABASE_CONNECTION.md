# Quick Fix: Database Connection Error

## Current Problem
```
"host": "localhost"  ← This won't work on Heroku!
```

## Solution: Set Your Database Hostname

### Step 1: Find Your Database Hostname

Your database hostname is **NOT** `localhost`. It should be something like:
- `mysql.yourhosting.com`
- `123.45.67.89` (IP address)
- `your-db.xxxxx.rds.amazonaws.com` (if using AWS RDS)
- `your-database-host.com`

**Where to find it:**
1. Check your hosting provider's control panel (cPanel, etc.)
2. Look for "MySQL Hostname" or "Remote MySQL"
3. Check your database connection settings in your local PHP config

### Step 2: Update Heroku Config

Replace `localhost` with your actual database hostname:

```bash
heroku config:set DB_HOST=your-actual-database-host.com
```

**Examples:**
```bash
# If your host is mysql.yourhosting.com
heroku config:set DB_HOST=mysql.yourhosting.com

# If your host is an IP address
heroku config:set DB_HOST=123.45.67.89

# If using AWS RDS
heroku config:set DB_HOST=your-db.xxxxx.us-east-1.rds.amazonaws.com
```

### Step 3: Verify All Config

```bash
# Check current config
heroku config

# Should show:
# DB_HOST: your-actual-host.com  (NOT localhost)
# DB_USER: u520834156_uPAHOZone25
# DB_PASSWORD: Y+;a+*1y
# DB_NAME: u520834156_dbUPAHOZoning
```

### Step 4: Test Connection

```bash
# Test the connection
curl https://your-app-name.herokuapp.com/api/check
```

Should now show:
```json
{
  "database": {
    "connected": true,
    "records_available": 1234,
    "host": "your-actual-host.com"
  }
}
```

## If You Don't Know Your Database Hostname

### Option A: Check Your Local PHP Config

Look in your local PHP files or config for database connection:
```php
// Look for something like:
$host = "mysql.yourhosting.com";  // This is what you need!
```

### Option B: Check Your Hosting Provider

1. Log into your hosting control panel
2. Go to "MySQL Databases" or "Database" section
3. Look for "Hostname" or "Server" - this is your DB_HOST

### Option C: Contact Your Hosting Provider

Ask them: "What is the MySQL hostname for remote connections?"

## Common Database Hostname Formats

- **Shared Hosting**: `mysql.yourdomain.com` or `mysql.yourhosting.com`
- **cPanel**: Usually `localhost` for local, but remote is different
- **AWS RDS**: `your-db.xxxxx.region.rds.amazonaws.com`
- **Google Cloud SQL**: `your-ip-address` or `your-instance-name`
- **Azure**: `your-server.database.windows.net`

## Important Notes

⚠️ **Your database must allow remote connections:**
- Enable "Remote MySQL" in your hosting panel
- Add Heroku IPs to allowed hosts (or allow all for testing)
- Check firewall allows port 3306

## Complete Example

```bash
# 1. Set the correct hostname
heroku config:set DB_HOST=mysql.yourhosting.com

# 2. Verify
heroku config:get DB_HOST
# Should output: mysql.yourhosting.com

# 3. Test
curl https://your-app-name.herokuapp.com/api/check

# 4. If connected, train models
curl https://your-app-name.herokuapp.com/api/train
```

## Still Having Issues?

If you can't find your database hostname:
1. Check your local `config.json` file (if you have one)
2. Check your PHP database connection code
3. Contact your hosting provider support
4. Consider using AWS RDS or another cloud database service

