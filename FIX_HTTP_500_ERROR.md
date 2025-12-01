# Fix HTTP 500 Error - Troubleshooting Guide

## 🔴 Problem
You're getting an HTTP 500 error when accessing your PHP file on the live server:
```
upahozoning.bccbsis.com is currently unable to handle this request.
HTTP ERROR 500
```

## 🔍 Step-by-Step Diagnosis

### Step 1: Run the Diagnostic Test

1. Upload `test_api.php` to your server
2. Access it in your browser: `https://upahozoning.bccbsis.com/test_api.php`
3. This will show you exactly what's wrong

### Step 2: Check Common Issues

#### Issue 1: cURL Not Enabled
**Symptoms:** Test shows "cURL is NOT enabled"

**Solution:**
- Contact your hosting provider to enable the cURL extension
- Or add this to your `.htaccess` or `php.ini`:
  ```ini
  extension=curl
  ```

#### Issue 2: PHP Version Too Old
**Symptoms:** Syntax errors or function not found

**Solution:**
- PHP 7.4 or higher is required
- Check your PHP version in `test_api.php`
- Contact hosting provider to upgrade if needed

#### Issue 3: SSL Certificate Issues
**Symptoms:** "SSL connection error" or "SSL certificate problem"

**Solution A (Recommended):**
- Update your server's CA certificates
- Contact your hosting provider

**Solution B (Temporary - NOT for production):**
- Edit `predict_land_cost_api.php`
- Find these lines (around line 80-81):
  ```php
  CURLOPT_SSL_VERIFYPEER => true,
  CURLOPT_SSL_VERIFYHOST => 2,
  ```
- Change to:
  ```php
  CURLOPT_SSL_VERIFYPEER => false,
  CURLOPT_SSL_VERIFYHOST => false,
  ```

#### Issue 4: File Permissions
**Symptoms:** "Permission denied" or file not readable

**Solution:**
```bash
# Set correct permissions
chmod 644 predict_land_cost_api.php
chmod 644 land_cost_prediction_ui.html
chmod 644 test_api.php
```

#### Issue 5: PHP Syntax Error
**Symptoms:** "Parse error" or syntax errors

**Solution:**
1. Check `test_api.php` output for syntax errors
2. Make sure you uploaded the complete file
3. Check for any special characters that got corrupted during upload

#### Issue 6: Memory Limit
**Symptoms:** "Fatal error: Allowed memory size exhausted"

**Solution:**
Add to the top of `predict_land_cost_api.php`:
```php
ini_set('memory_limit', '256M');
```

#### Issue 7: Timeout Issues
**Symptoms:** "Request timed out"

**Solution:**
The file already has a 60-second timeout. If still timing out:
- Check if your Heroku API is responding: https://upaho-883f1ffc88a8.herokuapp.com
- Check server's `max_execution_time` setting

## 🛠️ Quick Fixes

### Fix 1: Enable Error Display (Temporary)

Add this to the top of `predict_land_cost_api.php` (REMOVE after fixing):
```php
ini_set('display_errors', 1);
error_reporting(E_ALL);
```

This will show you the exact error. **Remove this after fixing!**

### Fix 2: Check Server Error Logs

1. Access your server's error log (cPanel, Plesk, or SSH)
2. Look for PHP errors related to `predict_land_cost_api.php`
3. Common log locations:
   - `/var/log/apache2/error.log`
   - `/var/log/php_errors.log`
   - `~/logs/error_log`
   - Check cPanel error logs

### Fix 3: Test API Directly

Test if your Heroku API is working:
```bash
curl -X POST https://upaho-883f1ffc88a8.herokuapp.com/predict/land_cost_future \
  -H "Content-Type: application/json" \
  -d '{
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

## 📋 Checklist

Before contacting support, check:

- [ ] `test_api.php` shows all tests passing
- [ ] cURL extension is enabled
- [ ] PHP version is 7.4 or higher
- [ ] File permissions are correct (644)
- [ ] Both files are uploaded completely
- [ ] No syntax errors in PHP file
- [ ] Heroku API is accessible
- [ ] Server error logs checked

## 🔧 Manual Testing

### Test 1: Check if PHP file is accessible
```
https://upahozoning.bccbsis.com/predict_land_cost_api.php
```
Should return JSON (even if error, should be JSON, not HTML)

### Test 2: Check if HTML file is accessible
```
https://upahozoning.bccbsis.com/land_cost_prediction_ui.html
```
Should show the form

### Test 3: Test with curl (from command line)
```bash
curl -X POST https://upahozoning.bccbsis.com/predict_land_cost_api.php \
  -H "Content-Type: application/json" \
  -d '{"prediction_type":"land_cost_future","target_years":10,"data":{"lot_area":200,"project_area":150,"project_type":"residential","location":"Downtown","year":2024,"month":1,"age":35}}'
```

## 🆘 Still Not Working?

### Contact Your Hosting Provider

Provide them with:
1. The error from `test_api.php`
2. Server error logs
3. PHP version
4. List of enabled PHP extensions

### Common Hosting Issues

**Shared Hosting:**
- May have restrictions on cURL
- May block outbound HTTPS connections
- May have firewall rules

**VPS/Dedicated:**
- Check firewall rules
- Check SELinux settings (if applicable)
- Check PHP-FPM/Apache configuration

## ✅ Success Indicators

When everything is working:
- `test_api.php` shows all green checkmarks
- `predict_land_cost_api.php` returns JSON (not HTML error)
- Form submission works and shows results
- No errors in browser console (F12)

## 📝 Notes

- The updated `predict_land_cost_api.php` has better error handling
- All errors now return JSON (not HTML)
- Better error messages for debugging
- Increased timeout to 60 seconds
- Handles SSL errors gracefully


