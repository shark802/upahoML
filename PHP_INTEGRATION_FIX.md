# PHP Integration Fix for SARIMAX Analytics

## Problem
The PHP interface was showing errors:
- "Error loading project types"
- "Failed to collect data: Unexpected token '<', "<br /><b>"... is not valid JSON"

This happens when PHP receives HTML error messages instead of JSON from Python scripts.

## Solution
All Python scripts have been updated to:
1. **Always output valid JSON** - No warnings, errors, or extra text
2. **Suppress stderr** - Prevents error messages from appearing in output
3. **Proper error handling** - Returns JSON error responses instead of exceptions

## Fixed Scripts

### 1. `get_project_types_api.py`
**Purpose:** Get list of project types for dropdown

**Usage in PHP:**
```php
$script_path = __DIR__ . "/../machine learning/predictive_analytics/get_project_types_api.py";
$command = "python " . escapeshellarg($script_path) . " 2>&1";
$output = shell_exec($command);
$data = json_decode($output, true);

if ($data && $data['success']) {
    foreach ($data['project_types'] as $type) {
        echo "<option value='{$type['id']}'>{$type['name']}</option>";
    }
} else {
    $error = $data['error'] ?? 'Unknown error';
    // Handle error
}
```

**Response Format:**
```json
{
  "success": true,
  "project_types": [
    {
      "id": 1,
      "name": "Residential",
      "category": "residential",
      "has_time_series_data": true,
      "time_series_count": 38
    }
  ],
  "total": 8
}
```

### 2. `collect_time_series_data.py`
**Purpose:** Collect/update time series data from application_forms

**Usage in PHP:**
```php
$script_path = __DIR__ . "/../machine learning/predictive_analytics/collect_time_series_data.py";
$command = "python " . escapeshellarg($script_path) . " 2>&1";
$output = shell_exec($command);
$data = json_decode($output, true);

if ($data && $data['success']) {
    echo "Collected: {$data['inserted']} inserted, {$data['updated']} updated";
} else {
    $error = $data['error'] ?? 'Unknown error';
    // Handle error
}
```

**Response Format:**
```json
{
  "success": true,
  "message": "Time series data collected successfully",
  "inserted": 16,
  "updated": 170,
  "total_records": 293
}
```

### 3. `sarimax_predictions.py`
**Purpose:** Generate SARIMAX forecasts

**Usage in PHP:**
```php
// Create input file
$input_data = [
    'project_type_id' => $_POST['project_type_id'],
    'metric_type' => $_POST['metric_type'] ?? 'application_count',
    'forecast_months' => $_POST['forecast_months'] ?? 12,
    'months_back' => $_POST['months_back'] ?? 24,
    'auto_select_order' => true
];

$input_file = tempnam(sys_get_temp_dir(), 'sarimax_');
file_put_contents($input_file, json_encode($input_data));

$script_path = __DIR__ . "/../machine learning/predictive_analytics/sarimax_predictions.py";
$command = "python " . escapeshellarg($script_path) . " " . escapeshellarg($input_file) . " 2>&1";
$output = shell_exec($command);

// Clean up
unlink($input_file);

$data = json_decode($output, true);
if ($data && $data['success']) {
    // Use $data['forecast'] for predictions
} else {
    $error = $data['error'] ?? 'Unknown error';
}
```

## Key Improvements

1. **Clean JSON Output**
   - All scripts output only JSON (no warnings, no errors)
   - Stderr is redirected to prevent error messages
   - Proper exception handling returns JSON errors

2. **Error Handling**
   - All errors return valid JSON with `success: false`
   - Error messages are in the `error` field
   - PHP can safely parse all responses

3. **Path Handling**
   - Scripts use relative paths that work from PHP
   - Config file is found automatically
   - No hardcoded paths

## Testing

Test each script directly:
```bash
# Test project types
python get_project_types_api.py

# Test data collection
python collect_time_series_data.py

# Test prediction (requires input file)
echo '{"project_type_id": 1, "metric_type": "application_count", "forecast_months": 12}' > test.json
python sarimax_predictions.py test.json
```

## Common Issues

### Issue: Still getting HTML errors
**Solution:** Make sure PHP is calling the script with `2>&1` to capture stderr:
```php
$command = "python " . escapeshellarg($script_path) . " 2>&1";
```

### Issue: Path not found
**Solution:** Use absolute paths or ensure PHP's working directory is correct:
```php
$script_path = realpath(__DIR__ . "/../machine learning/predictive_analytics/get_project_types_api.py");
```

### Issue: Permission denied
**Solution:** Ensure PHP has execute permissions on Python scripts:
```bash
chmod +x get_project_types_api.py
chmod +x collect_time_series_data.py
chmod +x sarimax_predictions.py
```

## Files Updated

1. ✅ `get_project_types_api.py` - Fixed JSON output, error handling
2. ✅ `collect_time_series_data.py` - New script for data collection
3. ✅ `sarimax_predictions.py` - Already had proper error handling

All scripts now guarantee valid JSON output for PHP integration.

