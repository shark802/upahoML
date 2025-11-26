# SARIMAX Project Types Fix

## Problem
The PHP interface shows "Error loading project types" when trying to use SARIMAX Analytics.

## Solution

### 1. Project Types API Script
Use `get_project_types_api.py` to get project types with their IDs from the `project_types` table:

```bash
python get_project_types_api.py
```

This returns JSON with project types and their corresponding IDs that match the `project_types` table (which is referenced by `application_time_series.project_type_id`).

**Example Output:**
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
    },
    ...
  ],
  "total": 8
}
```

### 2. PHP Integration
In your PHP file (`sarimax_analytics.php`), update the code to call this script:

```php
// Get project types
$script_path = __DIR__ . "/../machine learning/predictive_analytics/get_project_types_api.py";
$command = "python " . escapeshellarg($script_path);
$output = shell_exec($command);
$project_types_data = json_decode($output, true);

if ($project_types_data && $project_types_data['success']) {
    $project_types = $project_types_data['project_types'];
    // Populate dropdown
    foreach ($project_types as $type) {
        echo "<option value='{$type['id']}'>{$type['name']}</option>";
    }
} else {
    $error = $project_types_data['error'] ?? 'Unknown error';
    echo "Error loading project types: " . htmlspecialchars($error);
}
```

### 2. Time Series Data
The `application_time_series` table has been populated with aggregated monthly data from `application_forms`.

To repopulate/update:
```bash
python populate_time_series.py
```

### 3. PHP Integration
In your PHP file (`sarimax_analytics.php`), call the Python script to get project types:

```php
// Get project types
$command = "python " . escapeshellarg(__DIR__ . "/../machine learning/predictive_analytics/get_project_types_api.py");
$output = shell_exec($command);
$project_types_data = json_decode($output, true);

if ($project_types_data && $project_types_data['success']) {
    $project_types = $project_types_data['project_types'];
    // Use $project_types to populate dropdown
} else {
    // Handle error
    $error = $project_types_data['error'] ?? 'Unknown error';
}
```

### 4. Project Type Mapping
The IDs in the API response correspond to:
- `project_type_id` in `application_time_series` table
- Sequential IDs (1, 2, 3...) based on alphabetical order of project type names

### Available Project Types (from project_types table)
- Residential (ID: 1)
- Institutional (ID: 2)
- Commercial (ID: 3)
- Industrial (ID: 4)
- Agricultural (ID: 5)
- aquaculture (ID: 74)
- mangrove (ID: 75)
- parks_recreational (ID: 76)

**Note:** The IDs come from the `project_types` table, not sequential numbers.

## Testing

Test the SARIMAX prediction with a project type:

```bash
# Create test input (using ID from project_types table)
echo '{"project_type_id": 1, "metric_type": "application_count", "forecast_months": 12, "months_back": 24}' > test_sarimax.json

# Run prediction
python sarimax_predictions.py test_sarimax.json
```

**Important:** Use the IDs returned by `get_project_types_api.py`, not sequential numbers.

## Files Created/Updated

1. **get_project_types_api.py** - Main API script for PHP to get project types
2. **populate_time_series.py** - Script to populate/update application_time_series table
3. **setup_project_types_table.py** - Script to ensure project_types table exists
4. **map_project_types.py** - Helper script to see type mappings

