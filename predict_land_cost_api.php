<?php
/**
 * PHP Backend for Land Cost Prediction - API Version
 * Calls Flask API instead of Python script directly
 * 
 * Usage: Set HEROKU_API_URL environment variable or update $api_url below
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight OPTIONS request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

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

// Get API URL from environment or use default
// Option 1: Use Heroku app URL (set this in your PHP environment or config)
$api_url = getenv('HEROKU_API_URL') ?: 'https://your-app-name.herokuapp.com';

// Option 2: Use local Flask API if running locally
// $api_url = 'http://localhost:5000';

// Option 3: Use config file
// $config = json_decode(file_get_contents(__DIR__ . '/config.json'), true);
// $api_url = $config['api_url'] ?? 'https://your-app-name.herokuapp.com';

// Determine which endpoint to use based on prediction_type
$prediction_type = $data['prediction_type'] ?? 'land_cost_future';

if ($prediction_type === 'land_cost_future') {
    $endpoint = $api_url . '/predict/land_cost_future';
} elseif ($prediction_type === 'land_cost') {
    $endpoint = $api_url . '/predict/land_cost';
} else {
    // Use universal endpoint
    $endpoint = $api_url . '/api/predict';
}

// Prepare request data
$request_data = json_encode($data);

// Initialize cURL
$ch = curl_init($endpoint);

// Set cURL options
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => [
        'Content-Type: application/json',
        'Content-Length: ' . strlen($request_data)
    ],
    CURLOPT_POSTFIELDS => $request_data,
    CURLOPT_TIMEOUT => 30, // 30 second timeout
    CURLOPT_CONNECTTIMEOUT => 10, // 10 second connection timeout
    CURLOPT_SSL_VERIFYPEER => true, // Verify SSL certificate
    CURLOPT_SSL_VERIFYHOST => 2, // Verify hostname
]);

// Execute request
$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curl_error = curl_error($ch);

curl_close($ch);

// Handle cURL errors
if ($response === false || !empty($curl_error)) {
    echo json_encode([
        'success' => false,
        'error' => 'API request failed: ' . ($curl_error ?: 'Unknown error')
    ]);
    exit;
}

// Handle HTTP errors
if ($http_code !== 200) {
    $error_data = json_decode($response, true);
    echo json_encode([
        'success' => false,
        'error' => $error_data['error'] ?? "API returned HTTP $http_code"
    ]);
    exit;
}

// Parse and return response
$result = json_decode($response, true);

if ($result === null) {
    echo json_encode([
        'success' => false,
        'error' => 'Failed to parse API response. Raw: ' . substr($response, 0, 500)
    ]);
} else {
    echo json_encode($result);
}
?>

