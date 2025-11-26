<?php
/**
 * Updated PHP Backend for Land Cost Prediction
 * Now calls Flask API instead of Python script directly
 * 
 * This is an updated version of predict_land_cost.php that uses the Flask API
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

// ============================================
// CONFIGURATION: Set your Heroku API URL here
// ============================================
$api_url = 'https://your-app-name.herokuapp.com'; // CHANGE THIS to your Heroku app URL

// Or use environment variable:
// $api_url = getenv('HEROKU_API_URL') ?: 'https://your-app-name.herokuapp.com';

// Determine endpoint based on prediction type
$prediction_type = $data['prediction_type'] ?? 'land_cost_future';
$endpoint = $api_url . '/api/predict';

// Prepare request
$request_data = json_encode($data);

// Call Flask API using cURL
$ch = curl_init($endpoint);
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => [
        'Content-Type: application/json',
        'Content-Length: ' . strlen($request_data)
    ],
    CURLOPT_POSTFIELDS => $request_data,
    CURLOPT_TIMEOUT => 30,
    CURLOPT_CONNECTTIMEOUT => 10,
    CURLOPT_SSL_VERIFYPEER => true,
    CURLOPT_SSL_VERIFYHOST => 2,
]);

$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curl_error = curl_error($ch);
curl_close($ch);

// Handle errors
if ($response === false || !empty($curl_error)) {
    echo json_encode([
        'success' => false,
        'error' => 'API request failed: ' . ($curl_error ?: 'Unknown error')
    ]);
    exit;
}

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
        'error' => 'Failed to parse API response'
    ]);
} else {
    // Return the same format as before
    echo json_encode($result);
}
?>

