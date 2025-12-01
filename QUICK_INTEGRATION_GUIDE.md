# Quick Integration Guide - PHP Frontend with Heroku API

## ✅ What's Been Set Up

Your PHP frontend is now configured to integrate with the Heroku API at:
**https://upaho-883f1ffc88a8.herokuapp.com**

## 📁 Files Updated

1. **`predict_land_cost_api.php`** - Updated with your Heroku API URL
2. **`land_cost_prediction_ui.html`** - Updated to call the PHP backend instead of API directly

## 🚀 How to Use

### Step 1: Place Files on Your Server

Make sure both files are in your web server directory:
```
/htdocs/upaho_ml/upahoML/
├── predict_land_cost_api.php
└── land_cost_prediction_ui.html
```

### Step 2: Access the Frontend

Open in your browser:
```
http://localhost/upaho_ml/upahoML/land_cost_prediction_ui.html
```

Or if using XAMPP:
```
http://localhost/upahoML/land_cost_prediction_ui.html
```

### Step 3: Test the Integration

1. Fill out the form with land details
2. Click "🔮 Predict Land Cost"
3. The form will:
   - Send data to `predict_land_cost_api.php`
   - PHP will forward the request to your Heroku API
   - Results will be displayed in the UI

## 🔧 How It Works

```
User Form → JavaScript → predict_land_cost_api.php → Heroku API → Results
```

1. **Frontend (HTML)**: User fills form and submits
2. **JavaScript**: Sends POST request to `predict_land_cost_api.php`
3. **PHP Backend**: Receives request, forwards to Heroku API using cURL
4. **Heroku API**: Processes prediction and returns results
5. **PHP Backend**: Returns API response to frontend
6. **Frontend**: Displays results to user

## 📝 API Request Format

The PHP file sends this format to the Heroku API:

```json
{
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
}
```

## 🔄 Changing the API URL

If you need to change the API URL, edit `predict_land_cost_api.php`:

```php
// Line 34 - Update this line:
$api_url = getenv('HEROKU_API_URL') ?: 'https://upaho-883f1ffc88a8.herokuapp.com';
```

Or set an environment variable:
```php
putenv('HEROKU_API_URL=https://your-new-url.herokuapp.com');
```

## ✅ Testing

### Test the PHP Backend Directly

You can test the PHP file directly using curl:

```bash
curl -X POST http://localhost/upahoML/predict_land_cost_api.php \
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

### Test the Heroku API Directly

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

## 🐛 Troubleshooting

### Issue: "API request failed"
- Check that your Heroku app is running: Visit https://upaho-883f1ffc88a8.herokuapp.com
- Check PHP error logs
- Verify cURL is enabled in PHP: `php -m | grep curl`

### Issue: "Network error" in browser
- Open browser console (F12) to see detailed error
- Check that `predict_land_cost_api.php` is accessible
- Verify file permissions

### Issue: CORS errors
- The PHP file already includes CORS headers
- If issues persist, check that PHP is sending proper headers

### Issue: Timeout errors
- Increase timeout in `predict_land_cost_api.php`:
  ```php
  CURLOPT_TIMEOUT => 60, // Increase from 30 to 60 seconds
  ```

## 📊 Expected Response Format

The API returns:

```json
{
  "success": true,
  "prediction": {
    "current_prediction": {
      "cost_per_sqm": 34856.25,
      "total_value": 6971250.00
    },
    "future_prediction": {
      "target_year": 2034,
      "cost_per_sqm": 46175.50,
      "total_value": 9235100.00,
      "appreciation_rate": 0.0325,
      "total_appreciation": 0.325
    },
    "location_factors": {
      "category": "premium",
      "multiplier": 1.25,
      "description": "Premium Location"
    },
    "scenarios": {
      "optimistic": {...},
      "realistic": {...},
      "conservative": {...}
    },
    "yearly_breakdown": [...]
  }
}
```

## 🎯 Next Steps

1. ✅ Test the integration with sample data
2. ✅ Verify results are displayed correctly
3. ✅ Customize the UI if needed
4. ✅ Deploy to production server

## 📞 Support

If you encounter issues:
1. Check browser console (F12) for JavaScript errors
2. Check PHP error logs
3. Test the Heroku API directly to verify it's working
4. Verify all file paths are correct


