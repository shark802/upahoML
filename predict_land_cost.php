<?php
/**
 * PHP Backend for Land Cost Prediction
 * Calls Python prediction script and returns JSON
 * 
 * Supports:
 * - prediction_type: 'land_cost' (current year prediction)
 * - prediction_type: 'land_cost_future' (future prediction with ML)
 * - prediction_type: 'land_cost_arima' (ARIMA time series forecasting)
 * - use_arima: true (use ARIMA for future predictions)
 * 
 * Example ARIMA request:
 * {
 *   "prediction_type": "land_cost_arima",
 *   "data": {
 *     "lot_area": 200,
 *     "project_area": 150,
 *     "project_type": "residential",
 *     "location": "Downtown"
 *   },
 *   "target_years": 5,
 *   "use_arima": true
 * }
 */

header('Content-Type: application/json');

// Get JSON input
$input = file_get_contents('php://input');
$data = json_decode($input, true);

if (!$data) {
    echo json_encode([
        'success' => false,
        'error' => 'Invalid JSON input'
    ]);
    exit;
}

// Create temporary file for Python script
$temp_file = tempnam(sys_get_temp_dir(), 'land_pred_');
file_put_contents($temp_file, json_encode($data));

// Get script path
$script_path = __DIR__ . '/land_cost_predict.py';

// Build command
$command = 'python ' . escapeshellarg($script_path) . ' ' . escapeshellarg($temp_file) . ' 2>&1';

// Execute Python script
$output = shell_exec($command);

// Clean up temp file
unlink($temp_file);

// Clean output - remove any non-JSON text before the JSON
// Find the first { character which should be the start of JSON
$json_start = strpos($output, '{');
if ($json_start !== false) {
    $output = substr($output, $json_start);
}

// Parse output
$result = json_decode($output, true);

if ($result === null) {
    // If JSON decode failed, return error
    echo json_encode([
        'success' => false,
        'error' => 'Failed to parse prediction. Raw output: ' . substr($output, 0, 500)
    ]);
} else {
    echo json_encode($result);
}
?>

