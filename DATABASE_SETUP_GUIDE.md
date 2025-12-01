# Database Setup Guide for Heroku

## Problem: Error 2003 - Can't Connect to MySQL Server

**Cause**: Heroku apps cannot connect to `localhost` databases. You need a **cloud database**.

## Solution Options

### Option 1: Use Your Existing Remote MySQL Database (Recommended)

If you already have a remote MySQL database (like from a hosting provider):

1. **Get your database hostname** (not localhost):
   - Usually something like: `mysql.yourhost.com` or `123.45.67.89`
   - Check your hosting provider's database settings

2. **Set Heroku environment variables**:
   ```bash
   heroku config:set DB_HOST=your-database-host.com
   heroku config:set DB_USER=u520834156_uPAHOZone25
   heroku config:set DB_PASSWORD=Y+;a+*1y
   heroku config:set DB_NAME=u520834156_dbUPAHOZoning
   heroku config:set DB_PORT=3306
   ```

3. **Ensure database allows remote connections**:
   - Check your hosting provider allows remote MySQL connections
   - Whitelist Heroku IPs if needed (or allow all IPs for testing)

### Option 2: Use AWS RDS MySQL (Cloud Database)

1. **Create RDS MySQL instance**:
   - Go to AWS Console → RDS → Create Database
   - Choose MySQL
   - Set up security group to allow Heroku IPs
   - Note the endpoint (e.g., `your-db.xxxxx.us-east-1.rds.amazonaws.com`)

2. **Set Heroku config**:
   ```bash
   heroku config:set DB_HOST=your-db.xxxxx.us-east-1.rds.amazonaws.com
   heroku config:set DB_USER=your-rds-username
   heroku config:set DB_PASSWORD=your-rds-password
   heroku config:set DB_NAME=your-database-name
   heroku config:set DB_PORT=3306
   ```

### Option 3: Use Heroku Postgres (Requires Code Changes)

If you want to use Heroku Postgres, you'll need to:
1. Add Heroku Postgres addon
2. Modify code to use PostgreSQL instead of MySQL
3. Update connection code

**Not recommended** unless you want to migrate from MySQL.

## Quick Fix Steps

### Step 1: Check Current Config
```bash
heroku config
```

You'll see something like:
```
DB_HOST: localhost  ← This is the problem!
DB_USER: u520834156_uPAHOZone25
DB_PASSWORD: Y+;a+*1y
DB_NAME: u520834156_dbUPAHOZoning
```

### Step 2: Update DB_HOST

**If you have a remote database:**
```bash
heroku config:set DB_HOST=your-actual-database-host.com
```

**Example:**
```bash
heroku config:set DB_HOST=mysql.yourhosting.com
# or
heroku config:set DB_HOST=123.45.67.89
```

### Step 3: Verify Connection
```bash
curl https://your-app-name.herokuapp.com/api/check
```

Should show:
```json
{
  "success": true,
  "database": {
    "connected": true,
    "records_available": 1234
  }
}
```

## Finding Your Database Host

### If using shared hosting (cPanel, etc.):
1. Log into your hosting control panel
2. Go to MySQL Databases section
3. Look for "Remote MySQL" or "Hostname"
4. Usually shows: `mysql.yourdomain.com` or an IP address

### If using VPS/Dedicated server:
1. Use your server's public IP or domain
2. Ensure MySQL is configured to accept remote connections
3. Check firewall allows port 3306

### If using cloud provider:
- **AWS RDS**: Use the RDS endpoint
- **Google Cloud SQL**: Use the Cloud SQL IP
- **Azure Database**: Use the Azure server name

## Testing Database Connection

### From Heroku CLI:
```bash
heroku run python -c "
import mysql.connector
import os
conn = mysql.connector.connect(
    host=os.environ.get('DB_HOST'),
    user=os.environ.get('DB_USER'),
    password=os.environ.get('DB_PASSWORD'),
    database=os.environ.get('DB_NAME'),
    port=int(os.environ.get('DB_PORT', 3306))
)
print('Connection successful!')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM application_forms')
print(f'Records: {cursor.fetchone()[0]}')
conn.close()
"
```

## Common Issues

### Issue: "Access denied for user"
**Solution**: Check username and password are correct

### Issue: "Host is not allowed to connect"
**Solution**: 
- Enable remote MySQL access in your hosting panel
- Add Heroku IPs to allowed hosts
- Or allow all IPs (less secure, but works for testing)

### Issue: "Connection timeout"
**Solution**:
- Check database host is correct
- Verify database server is running
- Check firewall allows port 3306
- Ensure database allows connections from Heroku IPs

### Issue: Still using localhost
**Solution**: 
```bash
# Check current value
heroku config:get DB_HOST

# Update it
heroku config:set DB_HOST=your-actual-host.com

# Verify
heroku config:get DB_HOST
```

## Security Notes

⚠️ **Important**: 
- Never commit database credentials to Git
- Use Heroku config vars for all sensitive data
- Consider using SSL connections for production
- Restrict database access to specific IPs when possible

## Next Steps

1. ✅ Get your database hostname (not localhost)
2. ✅ Set `DB_HOST` environment variable
3. ✅ Test connection: `GET /api/check`
4. ✅ Train models: `GET /api/train`
5. ✅ Make predictions

## Example: Complete Setup

```bash
# 1. Set all database config
heroku config:set DB_HOST=mysql.yourhosting.com
heroku config:set DB_USER=u520834156_uPAHOZone25
heroku config:set DB_PASSWORD=Y+;a+*1y
heroku config:set DB_NAME=u520834156_dbUPAHOZoning
heroku config:set DB_PORT=3306

# 2. Verify config
heroku config

# 3. Test connection
curl https://your-app-name.herokuapp.com/api/check

# 4. Train models
curl https://your-app-name.herokuapp.com/api/train
```

## Need Help?

If you don't have a remote database:
1. Contact your hosting provider for remote MySQL access
2. Or set up AWS RDS (free tier available)
3. Or use a MySQL hosting service (like ClearDB, JawsDB)



