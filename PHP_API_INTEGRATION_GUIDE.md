# PHP to Flask API Integration Guide

## Overview

This guide shows how to update your PHP system to call the Flask API instead of running Python scripts directly.

## Files Created

1. **`predict_land_cost_api.php`** - New PHP file that calls Flask API
2. **Updated `app.py`** - Added `/api/predict` endpoint compatible with PHP format

## Setup Options

### Option 1: Use Heroku API (Recommended for Production)

1. **Update the API URL in PHP**:
   ```php
   // In predict_land_cost_api.php, set your Heroku app URL:
   $api_url = 'https://your-app-name.herokuapp.com';
   ```

2. **Or use environment variable**:
   ```php
   // Set in your PHP environment or .htaccess
   SetEnv HEROKU_API_URL https://your-app-name.herokuapp.com
   ```

### Option 2: Use Local Flask API (For Development)

1. **Run Flask locally**:
   ```bash
   python app.py
   # Runs on http://localhost:5000
   ```

2. **Update PHP to use local URL**:
   ```php
   $api_url = 'http://localhost:5000';
   ```

## Migration Steps

### Step 1: Update Your PHP Code

Replace your existing `predict_land_cost.php` with `predict_land_cost_api.php`, or update the existing file:

**Old code (calls Python directly):**
```php
$command = 'python ' . escapeshellarg($script_path) . ' ' . escapeshellarg($temp_file) . ' 2>&1';
$output = shell_exec($command);
```

**New code (calls Flask API):**
```php
$ch = curl_init($api_url . '/api/predict');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
    CURLOPT_POSTFIELDS => json_encode($data),
]);
$response = curl_exec($ch);
```

### Step 2: Test the API

Test the API endpoint directly:
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

### Step 3: Update Your Frontend

If your frontend calls `predict_land_cost.php`, it should work without changes since the response format is the same.

## API Endpoints

### 1. Universal Endpoint (PHP Compatible)
```
POST /api/predict
```

**Request Format:**
```json
{
  "prediction_type": "land_cost" | "land_cost_future",
  "data": {
    "lot_area": 200,
    "project_area": 150,
    "project_type": "residential",
    "location": "Downtown",
    "year": 2024,
    "month": 1,
    "age": 35
  },
  "target_years": 10
}
```

**Response Format:**
```json
{
  "success": true,
  "prediction": {
    "current_prediction": {...},
    "future_prediction": {...},
    "yearly_breakdown": [...],
    "location_factors": {...},
    "scenarios": {...}
  }
}
```

### 2. Direct Endpoints

- `POST /predict/land_cost` - Current year prediction
- `POST /predict/land_cost_future` - Future prediction (5-10 years)

## Configuration

### Set API URL via Environment Variable

**In `.htaccess` (Apache):**
```apache
SetEnv HEROKU_API_URL https://your-app-name.herokuapp.com
```

**In `php.ini` or via `putenv()`:**
```php
putenv('HEROKU_API_URL=https://your-app-name.herokuapp.com');
```

### Set API URL in Config File

Create `config.json`:
```json
{
  "api_url": "https://your-app-name.herokuapp.com"
}
```

## Error Handling

The PHP file handles:
- ✅ Invalid JSON input
- ✅ cURL errors
- ✅ HTTP errors (non-200 responses)
- ✅ API response parsing errors
- ✅ Timeout errors (30 seconds)

## Benefits of API Approach

1. **No Python Required on Server**: PHP server doesn't need Python installed
2. **Scalable**: Heroku handles scaling automatically
3. **Centralized**: All predictions go through one API
4. **Easier Updates**: Update models without touching PHP code
5. **Better Error Handling**: Structured error responses
6. **Monitoring**: Can monitor API usage and performance

## Troubleshooting

### Issue: "API request failed"
**Solution**: 
- Check API URL is correct
- Verify Heroku app is running: `heroku ps`
- Check firewall/network allows outbound HTTPS

### Issue: "No trained models available"
**Solution**: 
- Train models on Heroku or upload pre-trained models
- Check models directory exists and has model files

### Issue: "Connection timeout"
**Solution**:
- Increase timeout in PHP: `CURLOPT_TIMEOUT => 60`
- Check Heroku app is not sleeping (upgrade dyno if needed)

### Issue: SSL Certificate Error
**Solution**:
- For development, you can disable SSL verification (NOT recommended for production):
  ```php
  CURLOPT_SSL_VERIFYPEER => false,
  CURLOPT_SSL_VERIFYHOST => false,
  ```

## Testing

### Test from PHP:
```php
$test_data = [
    'prediction_type' => 'land_cost_future',
    'target_years' => 10,
    'data' => [
        'lot_area' => 200,
        'project_area' => 150,
        'project_type' => 'residential',
        'location' => 'Downtown',
        'year' => 2024,
        'month' => 1,
        'age' => 35
    ]
];

$ch = curl_init('https://your-app-name.herokuapp.com/api/predict');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
    CURLOPT_POSTFIELDS => json_encode($test_data),
]);
$response = curl_exec($ch);
echo $response;
```

## Next Steps

1. ✅ Update `predict_land_cost_api.php` with your Heroku URL
2. ✅ Test API endpoint directly
3. ✅ Replace old PHP file or update existing one
4. ✅ Test from your PHP application
5. ✅ Monitor Heroku logs: `heroku logs --tail`



