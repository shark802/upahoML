<?php
/**
 * PHP Backend for Land Cost Prediction - API Version
 * Calls Flask API instead of Python script directly
 * 
 * Usage: Set HEROKU_API_URL environment variable or update $api_url below
 */

// Enable error reporting for debugging (disable in production)
error_reporting(E_ALL);
ini_set('display_errors', 0); // Don't display errors, but log them
ini_set('log_errors', 1);

// Set headers first
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight OPTIONS request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

// Check if cURL is available
if (!function_exists('curl_init')) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'cURL extension is not enabled on this server. Please contact your hosting provider.'
    ]);
    exit;
}

try {
    // Get JSON input
    $input = file_get_contents('php://input');
    
    if ($input === false || empty($input)) {
        http_response_code(400);
        echo json_encode([
            'success' => false,
            'error' => 'No input data received'
        ]);
        exit;
    }
    
    $data = json_decode($input, true);
    
    if (!$data) {
        $json_error = json_last_error_msg();
        http_response_code(400);
        echo json_encode([
            'success' => false,
            'error' => 'Invalid JSON input: ' . $json_error
        ]);
        exit;
    }
    
    // Get API URL from environment or use default
    $api_url = getenv('HEROKU_API_URL');
    if (empty($api_url)) {
        $api_url = 'https://upaho-883f1ffc88a8.herokuapp.com';
    }
    
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
    
    if ($request_data === false) {
        http_response_code(500);
        echo json_encode([
            'success' => false,
            'error' => 'Failed to encode request data: ' . json_last_error_msg()
        ]);
        exit;
    }
    
    // Initialize cURL
    $ch = curl_init($endpoint);
    
    if ($ch === false) {
        http_response_code(500);
        echo json_encode([
            'success' => false,
            'error' => 'Failed to initialize cURL'
        ]);
        exit;
    }
    
    // Set cURL options
    $curl_options = [
        CURLOPT_POST => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
            'Content-Length: ' . strlen($request_data)
        ],
        CURLOPT_POSTFIELDS => $request_data,
        CURLOPT_TIMEOUT => 60, // Increased timeout to 60 seconds
        CURLOPT_CONNECTTIMEOUT => 15, // 15 second connection timeout
        CURLOPT_SSL_VERIFYPEER => true, // Verify SSL certificate
        CURLOPT_SSL_VERIFYHOST => 2, // Verify hostname
        CURLOPT_FOLLOWLOCATION => true, // Follow redirects
        CURLOPT_MAXREDIRS => 3, // Maximum redirects
    ];
    
    // If SSL verification fails, you can temporarily disable it (NOT recommended for production)
    // Uncomment these lines if you get SSL certificate errors:
    // $curl_options[CURLOPT_SSL_VERIFYPEER] = false;
    // $curl_options[CURLOPT_SSL_VERIFYHOST] = false;
    
    curl_setopt_array($ch, $curl_options);
    
    // Execute request
    $response = curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curl_error = curl_error($ch);
    $curl_errno = curl_errno($ch);
    
    curl_close($ch);
    
    // Handle cURL errors
    if ($response === false || !empty($curl_error)) {
        http_response_code(500);
        $error_message = 'API request failed';
        
        if ($curl_errno) {
            switch ($curl_errno) {
                case CURLE_COULDNT_CONNECT:
                    $error_message .= ': Could not connect to API server. Please check if the API is running.';
                    break;
                case CURLE_OPERATION_TIMEOUTED:
                    $error_message .= ': Request timed out. The API server may be slow or unavailable.';
                    break;
                case CURLE_SSL_CONNECT_ERROR:
                    $error_message .= ': SSL connection error. There may be an issue with the SSL certificate.';
                    break;
                default:
                    $error_message .= ': ' . $curl_error . ' (Error code: ' . $curl_errno . ')';
            }
        } else {
            $error_message .= ': ' . ($curl_error ?: 'Unknown error');
        }
        
        echo json_encode([
            'success' => false,
            'error' => $error_message,
            'debug' => [
                'endpoint' => $endpoint,
                'curl_errno' => $curl_errno,
                'curl_error' => $curl_error
            ]
        ]);
        exit;
    }
    
    // Handle HTTP errors
    if ($http_code !== 200) {
        http_response_code($http_code);
        $error_data = json_decode($response, true);
        
        $error_message = "API returned HTTP $http_code";
        if ($error_data && isset($error_data['error'])) {
            $error_message = $error_data['error'];
        } elseif (!empty($response)) {
            // Try to extract error from HTML response
            $error_message .= ': ' . substr(strip_tags($response), 0, 200);
        }
        
        echo json_encode([
            'success' => false,
            'error' => $error_message,
            'http_code' => $http_code,
            'response_preview' => substr($response, 0, 500)
        ]);
        exit;
    }
    
    // Parse and return response
    $result = json_decode($response, true);
    
    if ($result === null) {
        http_response_code(500);
        $json_error = json_last_error_msg();
        echo json_encode([
            'success' => false,
            'error' => 'Failed to parse API response: ' . $json_error,
            'response_preview' => substr($response, 0, 500)
        ]);
        exit;
    }
    
    // Return successful response
    http_response_code(200);
    echo json_encode($result);
    
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Server error: ' . $e->getMessage(),
        'file' => $e->getFile(),
        'line' => $e->getLine()
    ]);
} catch (Error $e) {
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Fatal error: ' . $e->getMessage(),
        'file' => $e->getFile(),
        'line' => $e->getLine()
    ]);
}
?>
