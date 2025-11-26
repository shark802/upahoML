# Land Cost Prediction UI - Setup Guide

## Files Created

1. **`land_cost_prediction_ui.html`** - Complete HTML interface with form and results display
2. **`predict_land_cost.php`** - PHP backend that calls Python prediction script

## Features

✅ **Beautiful, modern UI** with gradient design  
✅ **Responsive layout** - works on desktop and mobile  
✅ **Real-time predictions** - instant results after form submission  
✅ **Location-aware** - considers location premium/discount  
✅ **Multi-year projections** - 5 or 10 years ahead  
✅ **Multiple scenarios** - Optimistic, Realistic, Conservative  
✅ **Yearly breakdown** - See progression year by year  
✅ **No database required** - Pure prediction, no data storage  

## Setup Instructions

### 1. Place Files
Copy both files to your web server directory:
```
/path/to/your/website/
├── land_cost_prediction_ui.html
└── predict_land_cost.php
```

### 2. Ensure Python Script is Accessible
Make sure `land_cost_predict.py` is in the correct relative path:
- The PHP script looks for: `__DIR__ . '/land_cost_predict.py'`
- If your Python script is elsewhere, update the path in `predict_land_cost.php`

### 3. Set Permissions
```bash
chmod 644 land_cost_prediction_ui.html
chmod 755 predict_land_cost.php
```

### 4. Test
Open in browser:
```
http://localhost/land_cost_prediction_ui.html
```

## Usage

1. **Fill out the form:**
   - Lot Area (sqm)
   - Project Area (sqm)
   - Project Type (residential, commercial, etc.)
   - Location (Downtown, Urban Core, etc.)
   - Age (optional)
   - Project Years Ahead (5 or 10 years)

2. **Click "Predict Land Cost"**

3. **View results:**
   - Current year prediction
   - Future year prediction
   - Location category and multiplier
   - Three scenarios (optimistic, realistic, conservative)
   - Yearly breakdown

## Form Fields

| Field | Required | Description |
|-------|----------|-------------|
| Lot Area | Yes | Total lot size in square meters |
| Project Area | Yes | Building/development area in sqm |
| Project Type | Yes | Type of project (residential, commercial, etc.) |
| Location | Yes | Location of the land |
| Age | No | Age of applicant (default: 35) |
| Project Years Ahead | Yes | 5 or 10 years |

## Example Input

- **Lot Area**: 200 sqm
- **Project Area**: 150 sqm
- **Project Type**: Residential
- **Location**: Downtown
- **Age**: 35
- **Years Ahead**: 10

## Output Example

**Current (2025):**
- Cost: 34,856 PHP/sqm
- Total: 6,971,197 PHP

**Future (2035):**
- Cost: 46,175 PHP/sqm
- Total: 9,234,904 PHP
- Appreciation: 32.5% over 10 years

## Troubleshooting

### Issue: "Network error" or no response
**Solution**: 
- Check PHP error logs
- Verify Python script path in `predict_land_cost.php`
- Ensure Python is accessible from PHP
- Check file permissions

### Issue: "Failed to get prediction"
**Solution**:
- Verify `land_cost_predict.py` exists and is executable
- Check that models are trained (run training script)
- Verify database connection in Python script

### Issue: Results not displaying
**Solution**:
- Open browser console (F12) to check for JavaScript errors
- Verify PHP is returning valid JSON
- Check network tab for API response

## Customization

### Change Colors
Edit the CSS in `land_cost_prediction_ui.html`:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Add More Locations
Edit the location dropdown in the HTML:
```html
<option value="Your Location">Your Location</option>
```

### Change Default Values
Edit the form inputs:
```html
<input ... value="200">  <!-- Change default lot area -->
```

## Browser Compatibility

✅ Chrome/Edge (latest)  
✅ Firefox (latest)  
✅ Safari (latest)  
✅ Mobile browsers  

## Notes

- No data is saved to database
- All predictions are calculated in real-time
- Results are based on trained ML models
- Location factors are calculated from historical data

Enjoy your land cost prediction system! 🏠💰

